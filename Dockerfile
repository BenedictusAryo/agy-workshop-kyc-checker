# ==============================================================================
# AGY-KYC Checker - Production Dockerfile for Google Cloud Run
# ==============================================================================
FROM python:3.13-slim

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install system dependencies required for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Setup virtual environment
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN uv venv /app/.venv

# Copy compiled dependencies and install
COPY requirements.txt ./
RUN uv pip install --no-cache -r requirements.txt

# Copy application source code and install package
COPY pyproject.toml README.md .python-version ./
COPY app/ ./app/
COPY kyc_agent/ ./kyc_agent/
COPY tests/samples/ ./tests/samples/
RUN uv pip install --no-cache --no-deps -e .

# Create runtime directories
RUN mkdir -p data/uploads data/crops data/chroma

# Expose standard Cloud Run port
EXPOSE 8080

# Run FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
