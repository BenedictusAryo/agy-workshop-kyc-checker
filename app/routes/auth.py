"""Session authentication for Ops Agent Monitoring."""

import os
from typing import Optional
from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

router = APIRouter(tags=["Authentication"])
templates = Jinja2Templates(directory="app/templates")

SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "workshop-2026-session-secret-key-123")
COOKIE_NAME = "ops_session"
serializer = URLSafeTimedSerializer(SECRET_KEY)


def get_current_ops_user(request: Request) -> Optional[str]:
    """Extract and validate the logged-in Ops agent username from session cookie."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    try:
        data = serializer.loads(token, max_age=86400 * 7)  # 7 days
        return data.get("username")
    except (BadSignature, SignatureExpired):
        return None


def require_ops_auth(request: Request) -> str:
    """Dependency that requires an authenticated Ops user or redirects to login."""
    user = get_current_ops_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"},
        )
    return user


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Render the Ops Agent login page."""
    user = get_current_ops_user(request)
    if user:
        return RedirectResponse(url="/kyc-monitoring", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(request=request, name="login.html", context={"error": None})


@router.post("/login", response_class=HTMLResponse)
async def handle_login(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
):
    """Authenticate Ops Agent credentials and issue encrypted session cookie."""
    expected_user = os.getenv("OPS_USERNAME", "ops")
    expected_pass = os.getenv("OPS_PASSWORD", "workshop2026")

    if username.strip() == expected_user and password == expected_pass:
        token = serializer.dumps({"username": username.strip()})
        redirect = RedirectResponse(
            url="/kyc-monitoring", status_code=status.HTTP_303_SEE_OTHER
        )
        redirect.set_cookie(
            key=COOKIE_NAME,
            value=token,
            httponly=True,
            samesite="lax",
            max_age=86400 * 7,
        )
        return redirect

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": "Invalid Ops credentials. Default is: ops / workshop2026"},
        status_code=status.HTTP_401_UNAUTHORIZED,
    )



@router.get("/logout")
async def logout():
    """Clear session cookie and redirect to login."""
    redirect = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    redirect.delete_cookie(COOKIE_NAME)
    return redirect
