"""Chroma vector database integration for 1:N duplicate identity fraud detection."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings

logger = logging.getLogger("kyc_agent.vectordb")

CHROMA_PATH = Path("data/chroma")
COLLECTION_NAME = "kyc_face_embeddings"

_CLIENT = None
_COLLECTION = None


def get_chroma_collection():
    """Lazily initialize persistent local Chroma DB collection."""
    global _CLIENT, _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION

    os.makedirs(str(CHROMA_PATH), exist_ok=True)
    try:
        _CLIENT = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
        _COLLECTION = _CLIENT.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Connected to Chroma collection '{COLLECTION_NAME}' at {CHROMA_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize Chroma collection: {e}")
        _COLLECTION = None

    return _COLLECTION


def index_face(
    tracking_id: str,
    nik: Optional[str],
    nama: Optional[str],
    embedding: List[float],
) -> bool:
    """Store applicant face embedding into Chroma."""
    collection = get_chroma_collection()
    if collection is None or not embedding:
        return False

    try:
        clean_nik = str(nik or "").strip()
        clean_nama = str(nama or "").strip()
        collection.upsert(
            ids=[tracking_id],
            embeddings=[embedding],
            metadatas=[
                {
                    "tracking_id": tracking_id,
                    "nik": clean_nik,
                    "nama": clean_nama,
                }
            ],
            documents=[f"KYC Applicant {clean_nama} (NIK: {clean_nik})"],
        )
        return True
    except Exception as e:
        logger.warning(f"Error indexing face into Chroma: {e}")
        return False


def check_duplicate_identity(
    embedding: List[float],
    current_nik: Optional[str],
    current_tracking_id: str,
    similarity_threshold: float = 0.70,
) -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
    """
    Search Chroma for matching faces registered under a different NIK (Sybil / Identity Fraud).
    
    Returns:
        (is_duplicate, duplicate_explanation, matched_candidates)
    """
    collection = get_chroma_collection()
    if collection is None or not embedding:
        return False, None, []

    clean_nik = str(current_nik or "").strip()

    try:
        # Chroma cosine distance: distance = 1 - cosine_similarity
        # similarity >= 0.70 means distance <= 0.30
        max_distance = 1.0 - similarity_threshold
        results = collection.query(
            query_embeddings=[embedding],
            n_results=5,
            include=["metadatas", "distances"],
        )

        matched_records = []
        is_duplicate = False
        duplicate_reason = None

        if results and results.get("ids") and len(results["ids"]) > 0:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []

            for matched_id, meta, dist in zip(ids, metadatas, distances):
                if matched_id == current_tracking_id:
                    continue

                sim = max(0.0, 1.0 - dist)
                if sim >= similarity_threshold:
                    other_nik = str(meta.get("nik", "")).strip()
                    other_name = str(meta.get("nama", "UNKNOWN"))
                    matched_records.append(
                        {
                            "tracking_id": matched_id,
                            "nik": other_nik,
                            "nama": other_name,
                            "similarity": round(sim, 3),
                        }
                    )

                    # If the face matches another record that has a DIFFERENT NIK:
                    if clean_nik and other_nik and clean_nik != other_nik:
                        is_duplicate = True
                        duplicate_reason = (
                            f"Sybil Alert: This face was previously registered under a different NIK "
                            f"({other_nik[:6]}******{other_nik[-4:]}) by '{other_name}' "
                            f"(Tracking ID: {matched_id}) with {sim:.1%} similarity."
                        )

        return is_duplicate, duplicate_reason, matched_records
    except Exception as e:
        logger.warning(f"Error querying Chroma for duplicates: {e}")
        return False, None, []
