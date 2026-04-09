"""
routers/analyze.py
POST /api/v1/analyze — accept a conversation, extract beliefs, score it.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas import AnalysisResponse, ConversationRequest
from app.services.extractor import extract_beliefs
from app.services.ml_models import build_downstream_scores

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Analysis"])

DEBUG_MODE = os.getenv("DEBUG_RESPONSES", "false").lower() == "true"


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a conversation and extract user beliefs",
    description=(
        "Accepts a StoryBot conversation (messages_list + metadata), runs belief "
        "extraction via Gemma 4 on Ollama, applies ML scoring, and returns structured "
        "belief entries with downstream scores for three consumer teams."
    ),
    responses={
        200: {"description": "Successful analysis"},
        422: {"description": "Validation error in request payload"},
        503: {"description": "Ollama / model unavailable"},
    },
)
async def analyze_conversation(
    payload: ConversationRequest,
    debug: bool = Query(
        default=False,
        description="Include raw LLM output in the response (for development use only)",
    ),
) -> AnalysisResponse:
    """
    Main analysis endpoint.

    Steps:
    1. Validate incoming conversation payload (Pydantic).
    2. Extract beliefs via Gemma 4 (services.extractor).
    3. Compute downstream scores via ML layer (services.ml_models).
    4. Return structured AnalysisResponse.
    """
    include_raw = debug or DEBUG_MODE

    try:
        beliefs, meta = await extract_beliefs(payload, debug=include_raw)
    except Exception as exc:
        logger.exception(
            "Belief extraction failed for conversation %d: %s",
            payload.ref_conversation_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Belief extraction failed. Ensure Ollama is running and the "
                "gemma4:e4b model is pulled. "
                f"Detail: {exc}"
            ),
        ) from exc

    downstream = build_downstream_scores(beliefs, payload.messages_list, meta)

    user_messages = [m for m in payload.messages_list if m.ref_user_id != 1]

    return AnalysisResponse(
        ref_conversation_id=payload.ref_conversation_id,
        ref_user_id=payload.ref_user_id,
        analyzed_at=datetime.now(timezone.utc),
        message_count=len(payload.messages_list),
        user_message_count=len(user_messages),
        beliefs=beliefs,
        downstream_scores=downstream,
        raw_llm_output=meta.get("raw_llm_output") if include_raw else None,
    )
