"""
schemas.py
Pydantic v2 data models for the Belief Extraction API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ConversationMessage(BaseModel):
    """A single message row from the conversations.json format."""

    ref_conversation_id: int = Field(..., description="Conversation identifier")
    ref_user_id: int = Field(..., description="User identifier (1 == StoryBot)")
    transaction_datetime_utc: datetime = Field(..., description="UTC timestamp of the message")
    screen_name: str = Field(..., description="Display name of the sender")
    message: str = Field(..., description="Raw message text")

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


class ConversationRequest(BaseModel):
    """
    Payload POSTed to /api/v1/analyze.
    Accepts either a single conversation object or the full conversations.json
    array element structure.
    """

    messages_list: list[ConversationMessage] = Field(
        ..., min_length=1, description="Ordered list of conversation messages"
    )
    ref_conversation_id: int = Field(..., description="Conversation identifier")
    ref_user_id: int = Field(..., description="Primary user identifier (not StoryBot)")

    @field_validator("messages_list")
    @classmethod
    def has_user_messages(cls, messages: list[ConversationMessage]) -> list[ConversationMessage]:
        user_msgs = [m for m in messages if m.ref_user_id != 1]
        if not user_msgs:
            raise ValueError("messages_list must contain at least one non-StoryBot message")
        return messages


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class BeliefEntry(BaseModel):
    """A single extracted belief or self-perception from the conversation."""

    topic: str = Field(..., description="High-level topic area (e.g. 'health', 'family', 'autonomy')")
    belief: str = Field(..., description="The belief or perception the user holds")
    sentiment: str = Field(
        ..., description="Sentiment polarity: 'positive', 'negative', or 'neutral'"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence in this belief extraction (0–1)"
    )
    evidence: Optional[str] = Field(
        None, description="Brief quote or paraphrase from the conversation supporting this belief"
    )


class DownstreamScores(BaseModel):
    """Scores intended for the three downstream consumer teams."""

    # Team 1 – conversation value scoring
    conversation_value_score: float = Field(
        ..., ge=0.0, le=10.0,
        description="Overall richness / emotional depth of the conversation (0–10)"
    )
    # Team 2 – StoryBot next-turn guidance
    dominant_theme: str = Field(
        ..., description="Primary theme StoryBot should address in the next interaction"
    )
    suggested_follow_up: str = Field(
        ..., description="Suggested opening line or topic for StoryBot's next response"
    )
    # Team 3 – content recommendation tags
    content_tags: list[str] = Field(
        default_factory=list,
        description="Community content tags relevant to this user's beliefs"
    )


class AnalysisResponse(BaseModel):
    """Full response returned by POST /api/v1/analyze."""

    ref_conversation_id: int
    ref_user_id: int
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int
    user_message_count: int
    beliefs: list[BeliefEntry]
    downstream_scores: DownstreamScores
    raw_llm_output: Optional[Any] = Field(
        None, description="Raw Gemma output for debugging (omitted in production)"
    )
