"""Operations Agent KYC Monitoring and Verification Portal."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from kyc_agent.database import (
    get_submission,
    get_submissions_list,
    mask_nik,
    record_ops_decision,
)
from .auth import get_current_ops_user

router = APIRouter(tags=["Operations Monitoring"])
templates = Jinja2Templates(directory="app/templates")


class OpsReviewRequest(BaseModel):
    decision: str  # APPROVE, REJECT, RESUBMIT
    notes: Optional[str] = None


@router.get("/kyc-monitoring", response_class=HTMLResponse)
async def ops_dashboard_page(request: Request):
    """Render the Operations Agent KYC review dashboard."""
    user = get_current_ops_user(request)
    if not user:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    return templates.TemplateResponse(
        request=request,
        name="monitoring.html",
        context={"username": user},
    )



@router.get("/api/ops/submissions")
async def list_submissions_api(
    request: Request,
    filter: str = Query("review", enum=["review", "fraud", "verified", "all"]),
    limit: int = Query(100, ge=1, le=500),
):
    """API endpoint providing filtered submission queues for Ops."""
    user = get_current_ops_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not logged in")

    submissions = await get_submissions_list(filter_type=filter, limit=limit)
    return JSONResponse({"submissions": submissions, "filter": filter, "count": len(submissions)})


@router.get("/api/ops/submissions/{tracking_id}")
async def get_submission_detail_api(request: Request, tracking_id: str):
    """API returning comprehensive verification details for inspection modal."""
    user = get_current_ops_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not logged in")

    record = await get_submission(tracking_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found")

    # Generate public media URLs (handles both POSIX and Windows backslashes across platforms)
    from pathlib import Path

    ktp_path = record.get("ktp_path")
    selfie_path = record.get("selfie_path")
    crop_path = record.get("crop_path")

    record["ktp_url"] = f"/media/uploads/{Path(ktp_path.replace('\\', '/')).name}" if ktp_path else None
    record["selfie_url"] = f"/media/uploads/{Path(selfie_path.replace('\\', '/')).name}" if selfie_path else None
    record["crop_url"] = f"/media/crops/{Path(crop_path.replace('\\', '/')).name}" if crop_path else None

    return JSONResponse(record)


@router.post("/api/ops/review/{tracking_id}")
async def submit_ops_review_api(
    request: Request,
    tracking_id: str,
    payload: OpsReviewRequest,
):
    """Record an Ops Agent verification decision."""
    user = get_current_ops_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not logged in")

    if payload.decision not in ["APPROVE", "REJECT", "RESUBMIT"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Decision must be APPROVE, REJECT, or RESUBMIT",
        )

    ok = await record_ops_decision(
        tracking_id=tracking_id,
        decision=payload.decision,
        reviewer=user,
        notes=payload.notes,
    )

    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found")

    return JSONResponse(
        {
            "status": "success",
            "message": f"Submission {tracking_id} updated to {payload.decision}",
            "decision": payload.decision,
            "reviewer": user,
        }
    )
