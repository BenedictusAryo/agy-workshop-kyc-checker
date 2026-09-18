"""Applicant KYC upload and status tracking routes."""

import os
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from kyc_agent.agent import run_kyc_verification_pipeline
from kyc_agent.database import (
    create_submission,
    generate_tracking_id,
    get_submission,
    mask_name,
    mask_nik,
)

router = APIRouter(tags=["KYC"])
templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = Path("data/uploads")


@router.get("/", response_class=HTMLResponse)
@router.get("/kyc", response_class=HTMLResponse)
async def kyc_upload_page(request: Request):
    """Render the public KYC document upload interface."""
    return templates.TemplateResponse(request=request, name="kyc.html")


@router.get("/kyc-online", response_class=HTMLResponse)
async def kyc_online_page(request: Request):
    """Render the live webcam KYC capture interface."""
    return templates.TemplateResponse(request=request, name="kyc_online.html")



@router.post("/api/kyc/submit")
async def submit_kyc(
    background_tasks: BackgroundTasks,
    ktp: UploadFile = File(...),
    selfie: UploadFile = File(...),
):
    """Handle document uploads, generate 6-char tracking ID, and queue pipeline."""
    if not ktp.filename or not selfie.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both KTP and selfie images must be provided.",
        )

    # Validate image extensions
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}
    ktp_ext = Path(ktp.filename).suffix.lower()
    selfie_ext = Path(selfie.filename).suffix.lower()

    if ktp_ext not in allowed_exts or selfie_ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supported formats are JPG, JPEG, PNG, or WEBP.",
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    tracking_id = generate_tracking_id()

    ktp_path = UPLOAD_DIR / f"{tracking_id}_ktp{ktp_ext}"
    selfie_path = UPLOAD_DIR / f"{tracking_id}_selfie{selfie_ext}"

    # Write uploaded files to disk
    with open(ktp_path, "wb") as buffer:
        shutil.copyfileobj(ktp.file, buffer)

    with open(selfie_path, "wb") as buffer:
        shutil.copyfileobj(selfie.file, buffer)

    # Create SQLite record
    await create_submission(
        ktp_path=str(ktp_path),
        selfie_path=str(selfie_path),
        tracking_id=tracking_id,
    )

    # Queue Google ADK & Biometrics pipeline in background
    background_tasks.add_task(run_kyc_verification_pipeline, tracking_id)

    return JSONResponse(
        {
            "tracking_id": tracking_id,
            "status": "QUEUED",
            "redirect_url": f"/status/{tracking_id}",
            "message": "Submission received and queued for identity verification.",
        }
    )


@router.get("/status/{tracking_id}", response_class=HTMLResponse)
async def status_page(request: Request, tracking_id: str):
    """Render the applicant tracking page with masked identity fields."""
    record = await get_submission(tracking_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tracking ID not found.",
        )

    return templates.TemplateResponse(
        request=request,
        name="status.html",
        context={
            "tracking_id": tracking_id,
            "record": record,
            "masked_nik": mask_nik(record.get("nik")),
            "masked_nama": mask_name(record.get("nama")),
        },
    )



@router.get("/api/kyc/status/{tracking_id}")
async def get_kyc_status_api(tracking_id: str):
    """API endpoint polled by the frontend status page every 2.5s."""
    record = await get_submission(tracking_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tracking ID not found.",
        )

    ops_status = record.get("ops_status", "PENDING")
    completed = record.get("status") in ["VERIFIED", "MANUAL_REVIEW", "FRAUD"]

    # Determine customer-facing status: all completed submissions are IN_REVIEW until Ops decides
    if ops_status == "APPROVE":
        user_status = "APPROVED"
        customer_message = "Congratulations! Your identity has been successfully verified and approved."
    elif ops_status == "REJECT":
        user_status = "REJECTED"
        customer_message = "Your verification was not approved. Please contact support or resubmit with clearer documents."
    elif completed:
        user_status = "IN_REVIEW"
        customer_message = "Your documents have been received and automated analysis is complete. Our operations team is currently reviewing your application."
    else:
        user_status = "PROCESSING"
        customer_message = "Automated identity verification pipeline is in progress..."

    return JSONResponse(
        {
            "tracking_id": tracking_id,
            "status": record.get("status"),
            "ops_status": ops_status,
            "user_status": user_status,
            "masked_nik": mask_nik(record.get("nik")),
            "masked_nama": mask_name(record.get("nama")),
            "created_at": record.get("created_at"),
            "similarity_score": record.get("similarity_score"),
            "is_duplicate_face": bool(record.get("is_duplicate_face")),
            "ai_explanation": customer_message,
            "completed": completed,
        }
    )
