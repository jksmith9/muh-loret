"""
services/extractor.py
Feature extraction via Gemma 4 (gemma4:e4b) hosted on Ollama.

Responsibilities:
  1. Build a structured prompt from the user-only transcript.
  2. Call Ollama /api/generate asynchronously.
  3. Parse the model's JSON response into validated BeliefEntry objects.
  4. Return raw output for optional debug logging.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.schemas import BeliefEntry, ConversationRequest
from app.utils import (
    build_user_transcript,
    ollama_generate,
    safe_parse_json,
    truncate,
)

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert psychologist and NLP analyst specialising in belief elicitation.
Your task is to read a user's conversation transcript and identify the core beliefs
and self-perceptions the user holds. Output ONLY valid JSON — no prose, no markdown outside the JSON block.
"""

BELIEF_PROMPT_TEMPLATE = """\
Below is a conversation transcript between a user and StoryBot (an AI companion).
Only the user's messages are shown.

---
{transcript}
---

Analyse the transcript above and return a JSON object with the following exact structure:

{{
  "beliefs": [
    {{
      "topic": "<high-level topic: one of health, family, autonomy, technology, community, identity, leisure, spirituality, finances, other>",
      "belief": "<concise statement of what the user believes or feels about this topic>",
      "sentiment": "<positive | negative | neutral>",
      "confidence": <float between 0.0 and 1.0>,
      "evidence": "<short verbatim or paraphrased user quote supporting this belief>"
    }}
  ],
  "dominant_theme": "<the single most prominent theme in the conversation>",
  "suggested_follow_up": "<one sentence StoryBot could use to open the next interaction>",
  "content_tags": ["<tag1>", "<tag2>", "..."]
}}

Rules:
- Include 1–6 belief entries. Merge highly similar beliefs into one.
- Only extract beliefs clearly evidenced in the transcript.
- confidence reflects how clearly the belief is stated (1.0 = explicitly stated, 0.3 = inferred).
- content_tags should be lowercase kebab-case strings useful for content recommendation.
"""


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


async def extract_beliefs(
    conversation: ConversationRequest,
    debug: bool = False,
) -> tuple[list[BeliefEntry], dict[str, Any]]:
    """
    Run Gemma-4 belief extraction on a conversation.

    Returns:
        beliefs: List of validated BeliefEntry objects.
        meta:    Dict containing dominant_theme, suggested_follow_up, content_tags,
                 and (if debug=True) raw_llm_output.
    """
    messages_dicts = [m.model_dump(mode="json") for m in conversation.messages_list]
    transcript = build_user_transcript(messages_dicts)
    transcript = truncate(transcript, max_chars=4000)

    prompt = BELIEF_PROMPT_TEMPLATE.format(transcript=transcript)

    logger.info(
        "Extracting beliefs for conversation %d (user %d) via %s",
        conversation.ref_conversation_id,
        conversation.ref_user_id,
        OLLAMA_MODEL,
    )

    raw_output = await ollama_generate(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        prompt=prompt,
        system=SYSTEM_PROMPT,
    )

    parsed = safe_parse_json(raw_output)

    beliefs: list[BeliefEntry] = []
    meta: dict[str, Any] = {
        "dominant_theme": "unknown",
        "suggested_follow_up": "",
        "content_tags": [],
    }

    if parsed and isinstance(parsed, dict):
        meta["dominant_theme"] = parsed.get("dominant_theme", "unknown")
        meta["suggested_follow_up"] = parsed.get("suggested_follow_up", "")
        meta["content_tags"] = parsed.get("content_tags", [])

        for raw_belief in parsed.get("beliefs", []):
            try:
                beliefs.append(
                    BeliefEntry(
                        topic=raw_belief.get("topic", "other"),
                        belief=raw_belief.get("belief", ""),
                        sentiment=raw_belief.get("sentiment", "neutral"),
                        confidence=float(raw_belief.get("confidence", 0.5)),
                        evidence=raw_belief.get("evidence"),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed belief entry: %s — %s", raw_belief, exc)
    else:
        logger.error(
            "Failed to parse Gemma output for conversation %d. Raw: %.300s",
            conversation.ref_conversation_id,
            raw_output,
        )

    if debug:
        meta["raw_llm_output"] = raw_output

    return beliefs, meta
