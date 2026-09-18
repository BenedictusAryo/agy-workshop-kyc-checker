"""FastAPI application entrypoint for AGY-KYC Checker."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

from kyc_agent.database import init_db
from .routes import auth, kyc, ops


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan for database and directory initialization."""
    # Ensure runtime data directories exist
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/crops", exist_ok=True)
    os.makedirs("data/chroma", exist_ok=True)
    os.makedirs("app/static", exist_ok=True)

    # Initialize SQLite database schema
    await init_db()
    yield


app = FastAPI(
    title="AGY-KYC Checker",
    description="Indonesian e-KYC Verification using Google ADK, Gemini 3.8 Flash, AdaFace Biometrics, and Chroma",
    version="1.0.0",
    lifespan=lifespan,
)

# Ensure folders exist before mounting
os.makedirs("app/static", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/crops", exist_ok=True)

# Mount static and media directories
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/media/uploads", StaticFiles(directory="data/uploads"), name="media_uploads")
app.mount("/media/crops", StaticFiles(directory="data/crops"), name="media_crops")

# Include Routers
app.include_router(auth.router)
app.include_router(kyc.router)
app.include_router(ops.router)


@app.get("/healthz")
async def health_check():
    """Health check endpoint for Cloud Run and container orchestrators."""
    return {"status": "ok", "service": "agy-kyc-checker", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
