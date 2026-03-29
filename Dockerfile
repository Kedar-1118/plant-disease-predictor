# ──────────────────────────────────────────────
# PlantGuard AI — Dockerfile
# ──────────────────────────────────────────────
# Build:  docker build -t plantguard-ai .
# Run:    docker run -p 5000:5000 plantguard-ai
# ──────────────────────────────────────────────

FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    FLASK_DEBUG=false

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py config.py ./
COPY prediction/ prediction/
COPY utils/ utils/
COPY templates/ templates/
COPY static/css/ static/css/
COPY static/js/ static/js/
COPY data/treatments.json data/treatments.json

# Copy trained models
COPY models/ models/

# Create uploads directory
RUN mkdir -p static/uploads

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["python", "app.py"]
