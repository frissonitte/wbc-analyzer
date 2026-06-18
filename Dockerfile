# ── Stage 1: build ──────────────────────────────────────────────────────────
# Full image with build tools; installs compiled packages (numpy, opencv, tf).
FROM python:3.10-slim AS builder

WORKDIR /install

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --prefix=/install/deps -r requirements.txt

# ── Stage 2: runtime ────────────────────────────────────────────────────────
# Lean image; only compiled site-packages copied from builder.
FROM python:3.10-slim AS runtime

WORKDIR /app

# Runtime system libs opencv needs (no build tools).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder.
COPY --from=builder /install/deps /usr/local

# Copy application source (data/ excluded via .dockerignore except models).
COPY src/ src/
COPY docs/ docs/
COPY app.py .
COPY data/models/ data/models/

ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000
ENV FLASK_DEBUG=0
ENV FLASK_USE_RELOADER=0

EXPOSE 5000

CMD ["python", "app.py"]
