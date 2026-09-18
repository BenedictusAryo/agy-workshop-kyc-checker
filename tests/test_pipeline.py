"""Comprehensive integration and unit test suite for AGY-KYC Checker."""

import os
import shutil
import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport

from app.main import app
from kyc_agent.agent import run_kyc_verification_pipeline
from kyc_agent.biometrics import (
    compute_cosine_similarity,
    detect_and_crop_face,
    evaluate_biometrics,
)
from kyc_agent.database import (
    create_submission,
    get_submission,
    init_db,
    mask_name,
    mask_nik,
    record_ops_decision,
)
from kyc_agent.ocr import extract_ktp_ocr
from kyc_agent.vectordb import check_duplicate_identity, index_face

# Force test mock mode
os.environ["MOCK_MODE"] = "1"
TEST_DB = "data/test_kyc.db"


@pytest.fixture(autouse=True)
async def setup_test_env():
    """Setup clean test database before each test."""
    os.makedirs("data", exist_ok=True)
    await init_db(db_path=TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


def test_masking_functions():
    """Verify Indonesian privacy masking for NIK and full names."""
    assert mask_nik("3171021809950003") == "317102******0003"
    assert mask_nik(None) == "N/A"
    assert mask_nik("") == "N/A"

    assert mask_name("BUDI SANTOSO") == "B**I S*****O"
    assert mask_name("BUDI") == "B**I"
    assert mask_name(None) == "N/A"


@pytest.mark.asyncio
async def test_database_crud():
    """Verify SQLite submission lifecycle and audit trail."""
    tid = await create_submission(
        ktp_path="tests/samples/sample_ktp.jpg",
        selfie_path="tests/samples/sample_selfie.jpg",
        db_path=TEST_DB,
    )
    assert len(tid) == 6

    record = await get_submission(tid, db_path=TEST_DB)
    assert record is not None
    assert record["status"] == "QUEUED"
    assert record["ops_status"] == "PENDING"

    # Test Ops Agent review verdict
    updated = await record_ops_decision(
        tracking_id=tid,
        decision="APPROVE",
        reviewer="test_ops",
        notes="All documents verified cleanly.",
        db_path=TEST_DB,
    )
    assert updated is True

    record_after = await get_submission(tid, db_path=TEST_DB)
    assert record_after["status"] == "VERIFIED"
    assert record_after["ops_status"] == "APPROVE"
    assert record_after["ops_reviewer"] == "test_ops"


def test_ocr_extraction():
    """Verify Indonesian KTP extraction returns schema fields."""
    res = extract_ktp_ocr("tests/samples/sample_ktp.jpg")
    assert "nik" in res
    assert "nama" in res
    assert res["nik"] == "3171021809950003"
    assert res["nama"] == "BUDI SANTOSO"
    assert res["document_quality"] == "GOOD"
    assert res["is_valid_ktp"] is True
    assert isinstance(res["quality_reasons"], list)


def test_ocr_fraud_document():
    """Verify invalid or tampered document triggers is_valid_ktp=False."""
    res = extract_ktp_ocr("tests/samples/sample_ktp_fraud.jpg")
    assert res["is_valid_ktp"] is False
    assert res["document_quality"] == "TAMPERED"
    assert len(res["quality_reasons"]) > 0


@pytest.mark.asyncio
async def test_ops_windows_path_resolution():
    """Verify Windows backslash paths resolve properly to filenames across all OS platforms."""
    from pathlib import Path
    win_ktp = "data\\uploads\\WIN123_ktp.jpg"
    assert Path(win_ktp.replace("\\", "/")).name == "WIN123_ktp.jpg"



def test_biometrics_evaluation():
    """Verify face cropping and 1:1 AdaFace similarity calculation."""
    crop_path = "data/crops/test_crop.jpg"
    ok, path = detect_and_crop_face("tests/samples/sample_ktp.jpg", crop_path, is_ktp=True)
    assert ok is True
    assert os.path.exists(path)

    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert compute_cosine_similarity(v1, v2) == pytest.approx(1.0, 0.001)

    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert compute_cosine_similarity(v1, v3) == pytest.approx(0.0, 0.001)

    res = evaluate_biometrics(
        "tests/samples/sample_ktp.jpg",
        "tests/samples/sample_selfie.jpg",
        tracking_id="TEST01",
    )
    assert res["similarity_score"] > 0.0
    assert "biometric_verdict" in res


def test_chroma_duplicate_fraud():
    """Verify 1:N vector store identity deduplication."""
    face_emb = [0.1] * 512
    # Normalize
    norm = np.linalg.norm(face_emb)
    face_emb = (np.array(face_emb) / norm).tolist()

    # Index first identity
    idx_ok = index_face("ID0001", "3171011111110001", "ALICE", face_emb)
    assert idx_ok is True

    # Check same face with a different NIK -> Should trigger Sybil alert!
    is_dup, reason, _ = check_duplicate_identity(
        embedding=face_emb,
        current_nik="3171022222220002",
        current_tracking_id="ID0002",
        similarity_threshold=0.70,
    )
    assert is_dup is True
    assert "Sybil Alert" in reason


@pytest.mark.asyncio
async def test_api_full_journey():
    """Integration test: API endpoints, submission, polling, and Ops review."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Health check
        res = await ac.get("/healthz")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

        # 2. Public KYC page
        res = await ac.get("/kyc")
        assert res.status_code == 200

        # 3. Submit KYC documents
        with open("tests/samples/sample_ktp.jpg", "rb") as ktp_f, open(
            "tests/samples/sample_selfie.jpg", "rb"
        ) as selfie_f:
            files = {
                "ktp": ("ktp.jpg", ktp_f, "image/jpeg"),
                "selfie": ("selfie.jpg", selfie_f, "image/jpeg"),
            }
            res = await ac.post("/api/kyc/submit", files=files)
            assert res.status_code == 200
            data = res.json()
            assert "tracking_id" in data
            assert data["status"] == "QUEUED"
            tracking_id = data["tracking_id"]

        # Run pipeline directly for the created submission
        pipeline_res = await run_kyc_verification_pipeline(tracking_id)
        assert pipeline_res["status"] in ["VERIFIED", "MANUAL_REVIEW", "FRAUD"]

        # 4. Check applicant tracking status (should be IN_REVIEW before Ops action)
        res = await ac.get(f"/api/kyc/status/{tracking_id}")
        assert res.status_code == 200
        status_data = res.json()
        assert status_data["tracking_id"] == tracking_id
        assert status_data["completed"] is True
        assert status_data["masked_nik"] == "317102******0003"
        assert status_data["user_status"] == "IN_REVIEW"

        # 5. Ops portal without authentication -> Redirects
        res = await ac.get("/kyc-monitoring", follow_redirects=False)
        assert res.status_code in [302, 303, 307]

        # 6. Login to Ops
        login_res = await ac.post(
            "/login",
            data={"username": "ops", "password": "workshop2026"},
            follow_redirects=False,
        )
        assert login_res.status_code in [302, 303]
        ops_cookie = login_res.cookies.get("ops_session")
        assert ops_cookie is not None

        # 7. Authenticated Ops queue query
        ac.cookies.set("ops_session", ops_cookie)
        res = await ac.get("/api/ops/submissions?filter=all")
        assert res.status_code == 200
        subs = res.json()["submissions"]
        assert len(subs) > 0

        # 8. Ops decision execution
        review_res = await ac.post(
            f"/api/ops/review/{tracking_id}",
            json={"decision": "APPROVE", "notes": "Automated verification test pass."},
        )
        assert review_res.status_code == 200
        assert review_res.json()["decision"] == "APPROVE"

        # 9. Verify applicant status updates to APPROVED after Ops approval
        res = await ac.get(f"/api/kyc/status/{tracking_id}")
        assert res.status_code == 200
        approved_data = res.json()
        assert approved_data["ops_status"] == "APPROVE"
        assert approved_data["user_status"] == "APPROVED"


@pytest.mark.asyncio
async def test_kyc_online_endpoint():
    """Verify that /kyc-online renders the live webcam capture interface with step guides."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/kyc-online")
        assert res.status_code == 200
        assert "Live Camera" in res.text
        assert "webcamVideo" in res.text
        assert "Capture e-KTP Photo" in res.text
        assert "Capture Selfie Photo" in res.text
        assert "ktpGuideOverlay" in res.text
        assert "selfieGuideOverlay" in res.text
        assert "submitKycOnline" in res.text


