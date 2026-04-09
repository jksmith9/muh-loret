"""
tests/conftest.py
Shared pytest fixtures for the Belief Extraction API test suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient

# Ensure the build/app directory is on the Python path regardless of
# where pytest is invoked from.
BUILD_DIR = Path(__file__).parent.parent / "build"
sys.path.insert(0, str(BUILD_DIR))

from app.main import app  # noqa: E402


# ---------------------------------------------------------------------------
# TestClient
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """Synchronous TestClient wrapping the FastAPI app (no live Ollama needed)."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Sample conversation data
# ---------------------------------------------------------------------------

CONVERSATIONS_PATH = Path(__file__).parent.parent / "data" / "conversations.json"


@pytest.fixture(scope="session")
def all_conversations() -> list[dict[str, Any]]:
    """Load all conversations from the sample data file."""
    with CONVERSATIONS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def sample_conversation(all_conversations: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the first conversation from the sample data."""
    return all_conversations[0]


@pytest.fixture
def valid_payload(sample_conversation: dict[str, Any]) -> dict[str, Any]:
    """Return a well-formed ConversationRequest dict based on real sample data."""
    return {
        "messages_list": sample_conversation["messages_list"],
        "ref_conversation_id": sample_conversation["ref_conversation_id"],
        "ref_user_id": sample_conversation["ref_user_id"],
    }


@pytest.fixture
def minimal_payload() -> dict[str, Any]:
    """A minimal valid payload with exactly one user message."""
    return {
        "messages_list": [
            {
                "ref_conversation_id": 1,
                "ref_user_id": 42,
                "transaction_datetime_utc": "2024-01-01T10:00:00Z",
                "screen_name": "TestUser",
                "message": "I feel like I can handle anything life throws at me.",
            }
        ],
        "ref_conversation_id": 1,
        "ref_user_id": 42,
    }
