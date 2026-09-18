"""SQLite database and persistence layer for AGY-KYC Checker."""

import json
import os
import secrets
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiosqlite

DB_DIR = Path("data")
DB_PATH = DB_DIR / "kyc.db"


def generate_tracking_id(length: int = 6) -> str:
    """Generate a clean, readable 6-character random tracking ID."""
    alphabet = string.ascii_uppercase + string.digits
    # Remove easily confusable characters (0, O, 1, I)
    alphabet = alphabet.replace("0", "").replace("O", "").replace("1", "").replace("I", "")
    return "".join(secrets.choice(alphabet) for _ in range(length))


def mask_nik(nik: Optional[str]) -> str:
    """Mask Indonesian 16-digit NIK for public display (e.g., 317101******0002)."""
    if not nik:
        return "N/A"
    clean_nik = "".join(ch for ch in str(nik) if ch.isalnum())
    if len(clean_nik) < 8:
        return "***"
    prefix = clean_nik[:6]
    suffix = clean_nik[-4:]
    masked_len = max(len(clean_nik) - 10, 4)
    return f"{prefix}{'*' * masked_len}{suffix}"


def mask_name(name: Optional[str]) -> str:
    """Mask applicant name for public status display (e.g., B*** A***)."""
    if not name:
        return "N/A"
    words = name.strip().split()
    masked_words = []
    for word in words:
        if len(word) <= 1:
            masked_words.append(word)
        elif len(word) == 2:
            masked_words.append(word[0] + "*")
        else:
            masked_words.append(word[0] + "*" * (len(word) - 2) + word[-1])
    return " ".join(masked_words)


async def init_db(db_path: str = str(DB_PATH)):
    """Initialize the SQLite database schema."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                tracking_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                ktp_path TEXT,
                selfie_path TEXT,
                crop_path TEXT,
                nik TEXT,
                nama TEXT,
                tempat_lahir TEXT,
                tgl_lahir TEXT,
                jenis_kelamin TEXT,
                alamat TEXT,
                rt_rw TEXT,
                kel_desa TEXT,
                kecamatan TEXT,
                agama TEXT,
                status_perkawinan TEXT,
                pekerjaan TEXT,
                kewarganegaraan TEXT,
                ocr_json TEXT,
                similarity_score REAL,
                is_duplicate_face INTEGER DEFAULT 0,
                duplicate_notes TEXT,
                fraud_indicators TEXT,
                ai_explanation TEXT,
                ops_status TEXT DEFAULT 'PENDING',
                ops_reviewed_at TEXT,
                ops_reviewer TEXT,
                ops_notes TEXT
            );
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_status ON submissions(status);"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ops_status ON submissions(ops_status);"
        )
        await db.commit()


async def create_submission(
    ktp_path: str,
    selfie_path: str,
    tracking_id: Optional[str] = None,
    db_path: str = str(DB_PATH),
) -> str:
    """Create a new pending KYC submission record."""
    if not tracking_id:
        tracking_id = generate_tracking_id()

    now_iso = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO submissions (
                tracking_id, created_at, status, ktp_path, selfie_path, ops_status
            ) VALUES (?, ?, 'QUEUED', ?, ?, 'PENDING')
            """,
            (tracking_id, now_iso, ktp_path, selfie_path),
        )
        await db.commit()
    return tracking_id


async def update_submission_status(
    tracking_id: str,
    status: str,
    db_path: str = str(DB_PATH),
):
    """Update high-level processing status."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE submissions SET status = ? WHERE tracking_id = ?",
            (status, tracking_id),
        )
        await db.commit()


async def save_verification_results(
    tracking_id: str,
    status: str,
    crop_path: Optional[str],
    ocr_data: Dict[str, Any],
    similarity_score: float,
    is_duplicate_face: bool,
    duplicate_notes: Optional[str],
    fraud_indicators: List[str],
    ai_explanation: str,
    db_path: str = str(DB_PATH),
):
    """Store complete verification pipeline results."""
    nik = ocr_data.get("nik")
    nama = ocr_data.get("nama")
    tempat_lahir = ocr_data.get("tempat_lahir")
    tgl_lahir = ocr_data.get("tgl_lahir")
    jenis_kelamin = ocr_data.get("jenis_kelamin")
    alamat = ocr_data.get("alamat")
    rt_rw = ocr_data.get("rt_rw")
    kel_desa = ocr_data.get("kel_desa")
    kecamatan = ocr_data.get("kecamatan")
    agama = ocr_data.get("agama")
    status_perkawinan = ocr_data.get("status_perkawinan")
    pekerjaan = ocr_data.get("pekerjaan")
    kewarganegaraan = ocr_data.get("kewarganegaraan")
    ocr_json_str = json.dumps(ocr_data, ensure_ascii=False)
    fraud_str = json.dumps(fraud_indicators, ensure_ascii=False)

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            UPDATE submissions SET
                status = ?,
                crop_path = ?,
                nik = ?,
                nama = ?,
                tempat_lahir = ?,
                tgl_lahir = ?,
                jenis_kelamin = ?,
                alamat = ?,
                rt_rw = ?,
                kel_desa = ?,
                kecamatan = ?,
                agama = ?,
                status_perkawinan = ?,
                pekerjaan = ?,
                kewarganegaraan = ?,
                ocr_json = ?,
                similarity_score = ?,
                is_duplicate_face = ?,
                duplicate_notes = ?,
                fraud_indicators = ?,
                ai_explanation = ?
            WHERE tracking_id = ?
            """,
            (
                status,
                crop_path,
                nik,
                nama,
                tempat_lahir,
                tgl_lahir,
                jenis_kelamin,
                alamat,
                rt_rw,
                kel_desa,
                kecamatan,
                agama,
                status_perkawinan,
                pekerjaan,
                kewarganegaraan,
                ocr_json_str,
                similarity_score,
                1 if is_duplicate_face else 0,
                duplicate_notes,
                fraud_str,
                ai_explanation,
                tracking_id,
            ),
        )
        await db.commit()


async def get_submission(
    tracking_id: str,
    db_path: str = str(DB_PATH),
) -> Optional[Dict[str, Any]]:
    """Retrieve full details of a specific submission."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM submissions WHERE tracking_id = ?", (tracking_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            if res.get("ocr_json"):
                try:
                    res["ocr_data"] = json.loads(res["ocr_json"])
                except Exception:
                    res["ocr_data"] = {}
            else:
                res["ocr_data"] = {}

            if res.get("fraud_indicators"):
                try:
                    res["fraud_indicators_list"] = json.loads(res["fraud_indicators"])
                except Exception:
                    res["fraud_indicators_list"] = []
            else:
                res["fraud_indicators_list"] = []

            return res


async def get_submissions_list(
    filter_type: str = "review",
    limit: int = 100,
    db_path: str = str(DB_PATH),
) -> List[Dict[str, Any]]:
    """List submissions according to Ops filter queue."""
    query = "SELECT * FROM submissions "
    params: List[Any] = []

    if filter_type == "review":
        query += "WHERE status IN ('MANUAL_REVIEW', 'FRAUD') OR ops_status = 'PENDING' "
    elif filter_type == "fraud":
        query += "WHERE status = 'FRAUD' OR ops_status = 'REJECTED' "
    elif filter_type == "verified":
        query += "WHERE status = 'VERIFIED' OR ops_status = 'APPROVED' "
    elif filter_type == "all":
        pass  # No filter

    query += "ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, tuple(params)) as cursor:
            rows = await cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if item.get("ocr_json"):
                    try:
                        item["ocr_data"] = json.loads(item["ocr_json"])
                    except Exception:
                        item["ocr_data"] = {}
                else:
                    item["ocr_data"] = {}
                results.append(item)
            return results


async def record_ops_decision(
    tracking_id: str,
    decision: str,
    reviewer: str,
    notes: Optional[str] = None,
    db_path: str = str(DB_PATH),
) -> bool:
    """Record human Ops Agent review verdict (APPROVE, REJECT, RESUBMIT)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    status_map = {
        "APPROVE": "VERIFIED",
        "REJECT": "FRAUD",
        "RESUBMIT": "MANUAL_REVIEW",
    }
    new_system_status = status_map.get(decision.upper(), "MANUAL_REVIEW")

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            UPDATE submissions SET
                ops_status = ?,
                ops_reviewed_at = ?,
                ops_reviewer = ?,
                ops_notes = ?,
                status = ?
            WHERE tracking_id = ?
            """,
            (decision.upper(), now_iso, reviewer, notes or "", new_system_status, tracking_id),
        )
        await db.commit()
        return cursor.rowcount > 0
