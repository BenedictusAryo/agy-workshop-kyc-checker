"""Google ADK Agent definition and KYC pipeline orchestrator."""

import logging
import os
from typing import Any, Dict, List, Optional
from google.adk.agents.llm_agent import Agent

from .biometrics import evaluate_biometrics
from .database import (
    get_submission,
    save_verification_results,
    update_submission_status,
)
from .ocr import extract_ktp_ocr
from .vectordb import check_duplicate_identity, index_face

logger = logging.getLogger("kyc_agent.pipeline")


# Define ADK Agent Tools
def adk_extract_ktp_ocr(ktp_image_path: str) -> Dict[str, Any]:
    """ADK Tool: Extract Indonesian National ID (e-KTP) fields using Gemini 3.8 Flash.
    
    Args:
        ktp_image_path: Absolute or relative path to the KTP image file.
    """
    return extract_ktp_ocr(ktp_image_path)


def adk_evaluate_biometrics(
    ktp_image_path: str, selfie_image_path: str, tracking_id: str
) -> Dict[str, Any]:
    """ADK Tool: Compare KTP face with selfie using RetinaFace alignment and AdaFace ONNX embeddings.
    
    Args:
        ktp_image_path: Path to the KTP image.
        selfie_image_path: Path to the applicant selfie.
        tracking_id: Unique 6-character submission tracking identifier.
    """
    return evaluate_biometrics(ktp_image_path, selfie_image_path, tracking_id)


# Define Root ADK Agent for CLI / Web playground execution
root_agent = Agent(
    model=os.getenv("MODEL_ID", "gemini-3.8-flash"),
    name="agy_kyc_agent",
    description="Indonesian e-KYC Verification Agent for KTP OCR, Biometrics, and Fraud Detection.",
    instruction=(
        "You are the AGY-KYC Checker Agent. You verify Indonesian customer identities by: "
        "1. Extracting KTP details accurately via the 'adk_extract_ktp_ocr' tool. "
        "2. Performing face detection, cropping, and AdaFace biometric similarity via 'adk_evaluate_biometrics'. "
        "3. Synthesizing the final identity verification verdict (VERIFIED, MANUAL_REVIEW, or FRAUD)."
    ),
    tools=[adk_extract_ktp_ocr, adk_evaluate_biometrics],
)


async def run_kyc_verification_pipeline(tracking_id: str) -> Dict[str, Any]:
    """Execute complete 3-step verification workflow and record in SQLite and Chroma."""
    logger.info(f"Starting KYC verification pipeline for tracking_id: {tracking_id}")
    submission = await get_submission(tracking_id)
    if not submission:
        logger.error(f"Submission {tracking_id} not found in database.")
        return {"error": "Submission not found"}

    ktp_path = submission["ktp_path"]
    selfie_path = submission["selfie_path"]

    # Mark as PROCESSING
    await update_submission_status(tracking_id, "PROCESSING")

    # Step 1: KTP OCR via Gemini 3.8 Flash
    logger.info(f"[{tracking_id}] Step 1: Running KTP OCR...")
    ocr_result = extract_ktp_ocr(ktp_path)

    # Step 2 & 3: Face Crop & AdaFace Biometrics
    logger.info(f"[{tracking_id}] Step 2 & 3: Running Biometric evaluation...")
    bio_result = evaluate_biometrics(ktp_path, selfie_path, tracking_id)

    crop_path = bio_result.get("crop_path")
    sim_score = bio_result.get("similarity_score", 0.0)
    bio_verdict = bio_result.get("biometric_verdict", "MANUAL_REVIEW")
    selfie_emb = bio_result.get("selfie_embedding", [])

    # Step 4: Chroma 1:N Duplicate Identity Fraud Search
    nik = ocr_result.get("nik")
    nama = ocr_result.get("nama")
    logger.info(f"[{tracking_id}] Checking Chroma for duplicate identity fraud...")
    is_duplicate, dup_reason, _ = check_duplicate_identity(
        embedding=selfie_emb,
        current_nik=nik,
        current_tracking_id=tracking_id,
        similarity_threshold=0.72,
    )

    # Compile Fraud & Document Quality Indicators
    fraud_indicators: List[str] = []

    # 1. Non-KTP / Document Type Check
    if not ocr_result.get("is_valid_ktp", True) or ocr_result.get("document_quality") == "NOT_A_KTP":
        fraud_indicators.append("Invalid Document: Uploaded file does not depict an authentic Indonesian e-KTP card.")

    # 2. Duplicate Face / Sybil Fraud Alert
    if is_duplicate and dup_reason:
        fraud_indicators.append(dup_reason)

    # 3. Non-Facial Document & Physical Quality Observations
    if ocr_result.get("quality_reasons"):
        for q in ocr_result["quality_reasons"]:
            if q not in fraud_indicators:
                fraud_indicators.append(q)

    # 4. Digital Tampering / Formatting Warnings
    if ocr_result.get("validation_warnings"):
        for w in ocr_result["validation_warnings"]:
            if w not in fraud_indicators:
                fraud_indicators.append(w)

    # Determine Final Status
    is_doc_fraud = (
        not ocr_result.get("is_valid_ktp", True)
        or ocr_result.get("document_quality") in ["NOT_A_KTP", "TAMPERED"]
    )

    if is_doc_fraud:
        final_status = "FRAUD"
        if fraud_indicators:
            numbered_reasons = "\n".join([f"  {i+1}. {item}" for i, item in enumerate(fraud_indicators)])
            ai_explanation = f"Flagged as FRAUD (Document Integrity):\n{numbered_reasons}"
        else:
            ai_explanation = "Flagged as FRAUD (Document Integrity): Uploaded file does not depict an authentic Indonesian e-KTP card."
    elif is_duplicate:
        final_status = "FRAUD"
        ai_explanation = f"Flagged as FRAUD: Duplicate face detected under a different NIK. {dup_reason}"
    elif bio_verdict == "FRAUD":
        final_status = "FRAUD"
        ai_explanation = f"Flagged as FRAUD: Biometric mismatch ({sim_score:.1%}). {bio_result.get('reason')}"
    elif (
        bio_verdict == "MANUAL_REVIEW"
        or ocr_result.get("document_quality") in ["BLURRY", "GLARE", "CROPPED", "SUSPICIOUS"]
        or len(fraud_indicators) > 0
    ):
        final_status = "MANUAL_REVIEW"
        reasons_summary = "; ".join(fraud_indicators) if fraud_indicators else bio_result.get("reason")
        ai_explanation = f"Flagged for MANUAL REVIEW ({sim_score:.1%}): {reasons_summary}"
    else:
        final_status = "VERIFIED"
        ai_explanation = f"Successfully VERIFIED ({sim_score:.1%}): Valid e-KTP data and biometric match."

    # Index face into Chroma vector store for future Sybil detection
    if selfie_emb:
        index_face(tracking_id, nik, nama, selfie_emb)

    # Persist in SQLite
    await save_verification_results(
        tracking_id=tracking_id,
        status=final_status,
        crop_path=crop_path,
        ocr_data=ocr_result,
        similarity_score=sim_score,
        is_duplicate_face=is_duplicate,
        duplicate_notes=dup_reason,
        fraud_indicators=fraud_indicators,
        ai_explanation=ai_explanation,
    )

    logger.info(f"[{tracking_id}] Pipeline completed with status: {final_status}")
    return {
        "tracking_id": tracking_id,
        "status": final_status,
        "similarity_score": sim_score,
        "nik": nik,
        "nama": nama,
        "is_duplicate": is_duplicate,
        "explanation": ai_explanation,
    }
