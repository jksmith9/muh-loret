"""
tests/test_api.py
Integration tests for POST /api/v1/analyze using FastAPI TestClient.
Ollama is mocked via pytest-httpx so no live model is required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

BUILD_DIR = Path(__file__).parent.parent / "build"
sys.path.insert(0, str(BUILD_DIR))

from app.schemas import BeliefEntry, DownstreamScores

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MOCK_BELIEFS = [
    BeliefEntry(
        topic="health",
        belief="The user values their physical wellbeing.",
        sentiment="positive",
        confidence=0.88,
        evidence="I want to stay here as long as I can.",
    )
]

MOCK_META: dict[str, Any] = {
    "dominant_theme": "health",
    "suggested_follow_up": "How have you been feeling lately?",
    "content_tags": ["aging-in-place", "physical-wellbeing"],
}

MOCK_DOWNSTREAM = DownstreamScores(
    conversation_value_score=7.5,
    dominant_theme="health",
    suggested_follow_up="How have you been feeling lately?",
    content_tags=["aging-in-place", "physical-wellbeing"],
)


def _patch_services():
    """
    Context manager that mocks both extract_beliefs and build_downstream_scores
    so tests don't need a live Ollama or sklearn.
    """
    extract_patch = patch(
        "app.routers.analyze.extract_beliefs",
        new=AsyncMock(return_value=(MOCK_BELIEFS, MOCK_META)),
    )
    ml_patch = patch(
        "app.routers.analyze.build_downstream_scores",
        return_value=MOCK_DOWNSTREAM,
    )
    return extract_patch, ml_patch


SAMPLE_PAYLOAD: dict[str, Any] = {
    "messages_list": [
        {
            "ref_conversation_id": 42615,
            "ref_user_id": 822,
            "transaction_datetime_utc": "2023-10-01T08:01:00Z",
            "screen_name": "User822",
            "message": "I'm doing well, just trying to get through the day.",
        },
        {
            "ref_conversation_id": 42615,
            "ref_user_id": 822,
            "transaction_datetime_utc": "2023-10-02T09:00:00Z",
            "screen_name": "User822",
            "message": "I feel a bit isolated at home lately.",
        },
    ],
    "ref_conversation_id": 42615,
    "ref_user_id": 822,
}


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body

    def test_root_redirect(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/v1/analyze — happy path
# ---------------------------------------------------------------------------


class TestAnalyzeHappyPath:
    def test_returns_200_with_beliefs(self, client: TestClient) -> None:
        extract_patch, ml_patch = _patch_services()
        with extract_patch, ml_patch:
            resp = client.post("/api/v1/analyze", json=SAMPLE_PAYLOAD)

        assert resp.status_code == 200
        body = resp.json()
        assert body["ref_conversation_id"] == 42615
        assert body["ref_user_id"] == 822
        assert isinstance(body["beliefs"], list)
        assert len(body["beliefs"]) == 1
        assert body["beliefs"][0]["topic"] == "health"
        assert body["downstream_scores"]["conversation_value_score"] == 7.5

    def test_message_counts_correct(self, client: TestClient) -> None:
        extract_patch, ml_patch = _patch_services()
        with extract_patch, ml_patch:
            resp = client.post("/api/v1/analyze", json=SAMPLE_PAYLOAD)

        body = resp.json()
        assert body["message_count"] == 2
        assert body["user_message_count"] == 2

    def test_raw_llm_output_absent_by_default(self, client: TestClient) -> None:
        extract_patch, ml_patch = _patch_services()
        with extract_patch, ml_patch:
            resp = client.post("/api/v1/analyze", json=SAMPLE_PAYLOAD)

        assert resp.json()["raw_llm_output"] is None

    def test_debug_flag_returns_raw_output(self, client: TestClient) -> None:
        meta_with_raw = {**MOCK_META, "raw_llm_output": "raw gemma output here"}
        extract_patch = patch(
            "app.routers.analyze.extract_beliefs",
            new=AsyncMock(return_value=(MOCK_BELIEFS, meta_with_raw)),
        )
        ml_patch = patch(
            "app.routers.analyze.build_downstream_scores",
            return_value=MOCK_DOWNSTREAM,
        )
        with extract_patch, ml_patch:
            resp = client.post("/api/v1/analyze?debug=true", json=SAMPLE_PAYLOAD)

        assert resp.status_code == 200
        assert resp.json()["raw_llm_output"] == "raw gemma output here"

    def test_real_sample_data_validates(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        """Ensures the real data sample from data/conversations.json passes validation."""
        extract_patch, ml_patch = _patch_services()
        with extract_patch, ml_patch:
            resp = client.post("/api/v1/analyze", json=valid_payload)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/v1/analyze — validation errors
# ---------------------------------------------------------------------------


class TestAnalyzeValidationErrors:
    def test_empty_messages_list(self, client: TestClient) -> None:
        payload = {**SAMPLE_PAYLOAD, "messages_list": []}
        resp = client.post("/api/v1/analyze", json=payload)
        assert resp.status_code == 422

    def test_only_storybot_messages(self, client: TestClient) -> None:
        payload = {
            **SAMPLE_PAYLOAD,
            "messages_list": [
                {
                    "ref_conversation_id": 1,
                    "ref_user_id": 1,
                    "transaction_datetime_utc": "2024-01-01T10:00:00Z",
                    "screen_name": "StoryBot",
                    "message": "Hello! How can I help?",
                }
            ],
        }
        resp = client.post("/api/v1/analyze", json=payload)
        assert resp.status_code == 422

    def test_blank_message_text(self, client: TestClient) -> None:
        payload = {
            **SAMPLE_PAYLOAD,
            "messages_list": [
                {
                    "ref_conversation_id": 1,
                    "ref_user_id": 42,
                    "transaction_datetime_utc": "2024-01-01T10:00:00Z",
                    "screen_name": "User",
                    "message": "   ",
                }
            ],
        }
        resp = client.post("/api/v1/analyze", json=payload)
        assert resp.status_code == 422

    def test_missing_required_field(self, client: TestClient) -> None:
        payload = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "ref_conversation_id"}
        resp = client.post("/api/v1/analyze", json=payload)
        assert resp.status_code == 422

    def test_invalid_json_body(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/analyze",
            content="not json at all",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/analyze — Ollama failure
# ---------------------------------------------------------------------------


class TestAnalyzeOllamaFailure:
    def test_ollama_error_returns_503(self, client: TestClient) -> None:
        with patch(
            "app.routers.analyze.extract_beliefs",
            new=AsyncMock(side_effect=ConnectionError("Ollama not reachable")),
        ):
            resp = client.post("/api/v1/analyze", json=SAMPLE_PAYLOAD)

        assert resp.status_code == 503
        assert "Ollama" in resp.json()["detail"]
