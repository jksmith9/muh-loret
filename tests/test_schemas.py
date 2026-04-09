"""
tests/test_schemas.py
Unit tests for Pydantic models in app.schemas.
No network calls; all pure validation logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

BUILD_DIR = Path(__file__).parent.parent / "build"
sys.path.insert(0, str(BUILD_DIR))

from pydantic import ValidationError

from app.schemas import (
    AnalysisResponse,
    BeliefEntry,
    ConversationMessage,
    ConversationRequest,
    DownstreamScores,
)


# ---------------------------------------------------------------------------
# ConversationMessage
# ---------------------------------------------------------------------------


class TestConversationMessage:
    def test_valid_message(self) -> None:
        msg = ConversationMessage(
            ref_conversation_id=1,
            ref_user_id=42,
            transaction_datetime_utc="2024-01-01T10:00:00Z",
            screen_name="Alice",
            message="Hello there!",
        )
        assert msg.ref_user_id == 42
        assert msg.message == "Hello there!"

    def test_empty_message_rejected(self) -> None:
        with pytest.raises(ValidationError, match="message must not be blank"):
            ConversationMessage(
                ref_conversation_id=1,
                ref_user_id=42,
                transaction_datetime_utc="2024-01-01T10:00:00Z",
                screen_name="Alice",
                message="   ",
            )

    def test_whitespace_stripped(self) -> None:
        msg = ConversationMessage(
            ref_conversation_id=1,
            ref_user_id=42,
            transaction_datetime_utc="2024-01-01T10:00:00Z",
            screen_name="Alice",
            message="  trim me  ",
        )
        assert msg.message == "trim me"

    def test_missing_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ConversationMessage(  # type: ignore[call-arg]
                ref_conversation_id=1,
                ref_user_id=42,
                transaction_datetime_utc="2024-01-01T10:00:00Z",
                # screen_name missing
                message="Hello",
            )


# ---------------------------------------------------------------------------
# ConversationRequest
# ---------------------------------------------------------------------------


def _make_msg(user_id: int = 42, message: str = "Hi") -> dict:
    return {
        "ref_conversation_id": 100,
        "ref_user_id": user_id,
        "transaction_datetime_utc": "2024-06-01T08:00:00Z",
        "screen_name": "TestUser",
        "message": message,
    }


class TestConversationRequest:
    def test_valid_request(self) -> None:
        req = ConversationRequest(
            messages_list=[_make_msg()],
            ref_conversation_id=100,
            ref_user_id=42,
        )
        assert req.ref_conversation_id == 100
        assert len(req.messages_list) == 1

    def test_empty_messages_list_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ConversationRequest(
                messages_list=[],
                ref_conversation_id=100,
                ref_user_id=42,
            )

    def test_only_storybot_messages_rejected(self) -> None:
        """All messages from ref_user_id=1 (StoryBot) — should fail."""
        with pytest.raises(ValidationError, match="at least one non-StoryBot message"):
            ConversationRequest(
                messages_list=[_make_msg(user_id=1, message="Hi, I'm StoryBot!")],
                ref_conversation_id=100,
                ref_user_id=42,
            )

    def test_mixed_messages_accepted(self) -> None:
        req = ConversationRequest(
            messages_list=[
                _make_msg(user_id=1, message="Hello from StoryBot"),
                _make_msg(user_id=42, message="Hello from user"),
            ],
            ref_conversation_id=100,
            ref_user_id=42,
        )
        assert len(req.messages_list) == 2


# ---------------------------------------------------------------------------
# BeliefEntry
# ---------------------------------------------------------------------------


class TestBeliefEntry:
    def test_valid_belief(self) -> None:
        belief = BeliefEntry(
            topic="health",
            belief="I value my physical wellbeing.",
            sentiment="positive",
            confidence=0.85,
            evidence="I want to stay active and healthy.",
        )
        assert belief.confidence == 0.85

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            BeliefEntry(
                topic="health",
                belief="Test",
                sentiment="positive",
                confidence=1.5,  # > 1.0
            )

    def test_optional_evidence(self) -> None:
        belief = BeliefEntry(
            topic="family",
            belief="Family is my priority.",
            sentiment="positive",
            confidence=0.9,
        )
        assert belief.evidence is None


# ---------------------------------------------------------------------------
# DownstreamScores
# ---------------------------------------------------------------------------


class TestDownstreamScores:
    def test_valid_scores(self) -> None:
        scores = DownstreamScores(
            conversation_value_score=7.5,
            dominant_theme="health",
            suggested_follow_up="How have you been feeling this week?",
            content_tags=["mental-health", "aging-in-place"],
        )
        assert scores.conversation_value_score == 7.5
        assert "mental-health" in scores.content_tags

    def test_score_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            DownstreamScores(
                conversation_value_score=11.0,  # > 10
                dominant_theme="health",
                suggested_follow_up="How are you?",
            )

    def test_empty_tags_default(self) -> None:
        scores = DownstreamScores(
            conversation_value_score=5.0,
            dominant_theme="family",
            suggested_follow_up="Tell me more.",
        )
        assert scores.content_tags == []


# ---------------------------------------------------------------------------
# AnalysisResponse
# ---------------------------------------------------------------------------


class TestAnalysisResponse:
    def test_full_response(self) -> None:
        response = AnalysisResponse(
            ref_conversation_id=999,
            ref_user_id=42,
            analyzed_at=datetime.now(timezone.utc),
            message_count=10,
            user_message_count=5,
            beliefs=[
                BeliefEntry(
                    topic="health",
                    belief="I care about staying healthy.",
                    sentiment="positive",
                    confidence=0.9,
                )
            ],
            downstream_scores=DownstreamScores(
                conversation_value_score=8.0,
                dominant_theme="health",
                suggested_follow_up="How has your health been lately?",
                content_tags=["physical-wellbeing"],
            ),
        )
        assert response.ref_conversation_id == 999
        assert len(response.beliefs) == 1
        assert response.raw_llm_output is None
