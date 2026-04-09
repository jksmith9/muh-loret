"""
utils/__init__.py
Shared utilities: Ollama connectivity check, safe JSON parsing, text sanitisation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

STORYBOT_USER_ID = 1  # ref_user_id used by StoryBot in conversations.json


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------


async def check_ollama_health(base_url: str, model: str) -> bool:
    """
    Returns True if Ollama is reachable and the required model is available.
    Called during FastAPI lifespan startup.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            tags = resp.json()
            available = [m["name"] for m in tags.get("models", [])]
            if model not in available:
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s", model, available
                )
                return False
            return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Ollama health check failed: %s", exc)
        return False


async def ollama_generate(
    base_url: str,
    model: str,
    prompt: str,
    system: Optional[str] = None,
    timeout: float = 120.0,
) -> str:
    """
    Call Ollama's /api/generate endpoint and return the full response text.
    Uses stream=False for simplicity.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{base_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def extract_json_block(text: str) -> Optional[str]:
    """
    Extract the first JSON code-fence block (```json ... ```) from LLM output.
    Falls back to finding the first ``{`` … ``}`` span if no fence is present.
    """
    # Try fenced block first
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    # Fallback: find outermost braces
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def safe_parse_json(text: str) -> Optional[Any]:
    """
    Attempt to parse JSON from LLM output — handles fenced blocks & raw JSON.
    Returns None if parsing fails.
    """
    candidate = extract_json_block(text)
    if candidate is None:
        candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed for LLM output snippet: %.200s", text)
        return None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def build_user_transcript(messages: list[dict[str, Any]]) -> str:
    """
    Filter to user-only messages (exclude StoryBot) and format as a plain
    transcript for the LLM prompt.
    """
    lines = []
    for msg in messages:
        if msg.get("ref_user_id") == STORYBOT_USER_ID:
            continue
        name = msg.get("screen_name", "User")
        text = msg.get("message", "").strip()
        ts = msg.get("transaction_datetime_utc", "")
        lines.append(f"[{ts}] {name}: {text}")
    return "\n".join(lines)


def truncate(text: str, max_chars: int = 4000) -> str:
    """Safeguard: truncate transcript to stay within context window."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"
