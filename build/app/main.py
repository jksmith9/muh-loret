"""
main.py
FastAPI application entry-point for the Belief Extraction API.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.analyze import router as analyze_router
from app.utils import check_ollama_health

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (from env)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: RUF029
    """Check Ollama connectivity on startup; log result."""
    logger.info("Starting Belief Extraction API …")
    logger.info("Ollama base URL : %s", OLLAMA_BASE_URL)
    logger.info("Ollama model    : %s", OLLAMA_MODEL)

    healthy = await check_ollama_health(OLLAMA_BASE_URL, OLLAMA_MODEL)
    if healthy:
        logger.info("✓ Ollama is reachable and model '%s' is available.", OLLAMA_MODEL)
    else:
        logger.warning(
            "⚠  Ollama health check failed. Requests will fail until Ollama is "
            "running and '%s' is pulled (`ollama pull %s`).",
            OLLAMA_MODEL,
            OLLAMA_MODEL,
        )

    yield  # API is live

    logger.info("Shutting down Belief Extraction API.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Belief Extraction API",
    description=(
        "Analyse StoryBot conversation transcripts to extract user beliefs, "
        "self-perceptions, and topic attitudes using Gemma 4 via Ollama. "
        "Produces structured output for conversation scoring, StoryBot guidance, "
        "and community content recommendation."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(analyze_router)


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"], summary="API liveness check")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"message": "Belief Extraction API — visit /docs for the interactive API reference."}
