"""
tests/test_extractor.py
Unit tests for app.services.extractor using pytest-httpx to mock Ollama.
No live Ollama instance is required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
BUILD_DIR = Path(__file__).parent.parent / "build"
sys.path.insert(0, str(BUILD_DIR))

from pytest_httpx import HTTPXMock

from app.schemas import ConversationRequest
from app.services import extractor as extractor_module
from app.utils import build_user_transcript, extract_json_block, safe_parse_json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_OLLAMA_URL = "http://localhost:11434"

VALID_GEMMA_RESPONSE = {
    "beliefs": [
        {
            "topic": "health",
            "belief": "The user values physical safety and self-care.",
            "sentiment": "positive",
            "confidence": 0.9,
            "evidence": "I want to stay here for as long as I can.",
        },
        {
            "topic": "family",
            "belief": "The user relies on family for emotional support.",
            "sentiment": "positive",
            "confidence": 0.85,
            "evidence": "I'll talk to my daughter about it.",
        },
    ],
    "dominant_theme": "health",
    "suggested_follow_up": "How have you been feeling since your fall last week?",
    "content_tags": ["aging-in-place", "family-support"],
}


def _make_conversation(user_msgs: list[str]) -> ConversationRequest:
    messages = []
    for i, text in enumerate(user_msgs):
        messages.append(
            {
                "ref_conversation_id": 1,
                "ref_user_id": 42,
                "transaction_datetime_utc": f"2024-01-0{i + 1}T10:00:00Z",
                "screen_name": "TestUser",
                "message": text,
            }
        )
    return ConversationRequest(
        messages_list=messages,
        ref_conversation_id=1,
        ref_user_id=42,
    )


# ---------------------------------------------------------------------------
# Utils unit tests (no HTTP needed)
# ---------------------------------------------------------------------------


class TestBuildUserTranscript:
    def test_filters_storybot(self) -> None:
        messages = [
            {
                "ref_conversation_id": 1,
                "ref_user_id": 1,  # StoryBot
                "transaction_datetime_utc": "2024-01-01T10:00:00Z",
                "screen_name": "StoryBot",
                "message": "Hello, how are you?",
            },
            {
                "ref_conversation_id": 1,
                "ref_user_id": 42,
                "transaction_datetime_utc": "2024-01-01T10:01:00Z",
                "screen_name": "Alice",
                "message": "I'm doing well!",
            },
        ]
        transcript = build_user_transcript(messages)
        assert "StoryBot" not in transcript
        assert "I'm doing well!" in transcript

    def test_empty_returns_empty_string(self) -> None:
        assert build_user_transcript([]) == ""


class TestExtractJsonBlock:
    def test_fenced_block_extracted(self) -> None:
        text = 'Some preamble\n```json\n{"key": "value"}\n```\nSome suffix'
        assert extract_json_block(text) == '{"key": "value"}'

    def test_raw_json_extracted(self) -> None:
        text = 'Here is your answer: {"beliefs": []}'
        result = extract_json_block(text)
        assert result is not None
        assert "beliefs" in result

    def test_no_json_returns_none(self) -> None:
        assert extract_json_block("No JSON here at all.") is None


class TestSafeParseJson:
    def test_valid_json_parsed(self) -> None:
        result = safe_parse_json('{"beliefs": []}')
        assert result == {"beliefs": []}

    def test_invalid_json_returns_none(self) -> None:
        result = safe_parse_json("This is definitely not JSON!!!")
        assert result is None

    def test_fenced_json_parsed(self) -> None:
        text = "```json\n{\"key\": 1}\n```"
        result = safe_parse_json(text)
        assert result == {"key": 1}


# ---------------------------------------------------------------------------
# Extractor integration tests (Ollama mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_beliefs_success(httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: Ollama returns valid JSON, beliefs are parsed correctly."""
    monkeypatch.setattr(extractor_module, "OLLAMA_BASE_URL", MOCK_OLLAMA_URL)
    monkeypatch.setattr(extractor_module, "OLLAMA_MODEL", "gemma4:e4b")

    httpx_mock.add_response(
        url=f"{MOCK_OLLAMA_URL}/api/generate",
        method="POST",
        json={"response": f"```json\n{json.dumps(VALID_GEMMA_RESPONSE)}\n```"},
    )

    conversation = _make_conversation(
        ["I love spending time with my family.", "My health has been a concern lately."]
    )
    beliefs, meta = await extractor_module.extract_beliefs(conversation)

    assert len(beliefs) == 2
    assert beliefs[0].topic == "health"
    assert beliefs[1].topic == "family"
    assert meta["dominant_theme"] == "health"
    assert "aging-in-place" in meta["content_tags"]


@pytest.mark.asyncio
async def test_extract_beliefs_malformed_json(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ollama returns garbled output — extractor should return empty beliefs gracefully."""
    monkeypatch.setattr(extractor_module, "OLLAMA_BASE_URL", MOCK_OLLAMA_URL)
    monkeypatch.setattr(extractor_module, "OLLAMA_MODEL", "gemma4:e4b")

    httpx_mock.add_response(
        url=f"{MOCK_OLLAMA_URL}/api/generate",
        method="POST",
        json={"response": "I cannot determine any beliefs from this text."},
    )

    conversation = _make_conversation(["Hi."])
    beliefs, meta = await extractor_module.extract_beliefs(conversation)

    assert beliefs == []
    assert meta["dominant_theme"] == "unknown"


@pytest.mark.asyncio
async def test_extract_beliefs_debug_includes_raw(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With debug=True, raw_llm_output should be present in meta."""
    monkeypatch.setattr(extractor_module, "OLLAMA_BASE_URL", MOCK_OLLAMA_URL)
    monkeypatch.setattr(extractor_module, "OLLAMA_MODEL", "gemma4:e4b")

    raw = f"```json\n{json.dumps(VALID_GEMMA_RESPONSE)}\n```"
    httpx_mock.add_response(
        url=f"{MOCK_OLLAMA_URL}/api/generate",
        method="POST",
        json={"response": raw},
    )

    conversation = _make_conversation(["I feel confident about the future."])
    _, meta = await extractor_module.extract_beliefs(conversation, debug=True)

    assert "raw_llm_output" in meta
    assert meta["raw_llm_output"] == raw
