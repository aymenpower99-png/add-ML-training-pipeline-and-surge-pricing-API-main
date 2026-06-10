# ══════════════════════════════════════════════════════════════════
# Dockerfile — ML Pricing API (FastAPI)
# ══════════════════════════════════════════════════════════════════
# Service: Pricing API
# Port: 8000 (Railway injects $PORT)
# Entry: uvicorn main:app
# ══════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# ── System deps ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working dir ──
WORKDIR /app

# ── Install Python deps ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy app code ──
COPY . .

# ── Expose & run ──
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
