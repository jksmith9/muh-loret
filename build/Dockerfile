# ============================================================
# Belief Extraction API — Dockerfile
# Uses uv for fast, reproducible package installation.
# ============================================================

# ── Stage 1: builder ─────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Copy only the requirements file first (layer caching)
COPY app/requirements.txt .

# Install deps into an isolated prefix so we can copy them cleanly
RUN uv pip install --system --no-cache -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Copy installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application source
COPY app/ ./app/

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

# Expose API port
EXPOSE 8000

# Environment defaults (overridable at runtime)
ENV OLLAMA_BASE_URL="http://host.docker.internal:11434" \
    OLLAMA_MODEL="gemma4:e4b" \
    CORS_ORIGINS="*" \
    DEBUG_RESPONSES="false"

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
