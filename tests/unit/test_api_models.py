from datetime import datetime
from types import SimpleNamespace

from src.api.schemas import (
    RecallRequest,
    SearchRequest,
    TurnRequest,
    format_timestamp,
    format_user_memories,
)
from src.ingestion.intake import IngestTurnCommand, TurnMessage
from src.recall import RecallContextCommand
from src.core.search import SearchMemoriesCommand


def test_turn_request_builds_ingest_command():
    request = TurnRequest(
        session_id="session-1",
        user_id="user-1",
        messages=[
            {"role": "user", "content": "I work at Stripe."},
            {"role": "tool", "content": "{}", "name": "lookup"},
        ],
        timestamp="2025-05-01T10:00:00Z",
        metadata={"source": "test"},
    )

    assert request.to_command() == IngestTurnCommand(
        session_id="session-1",
        user_id="user-1",
        messages=[
            TurnMessage(role="user", content="I work at Stripe.", name=None),
            TurnMessage(role="tool", content="{}", name="lookup"),
        ],
        timestamp="2025-05-01T10:00:00Z",
        metadata={"source": "test"},
    )


def test_recall_request_builds_recall_command():
    request = RecallRequest(
        query="Where does the user work?",
        session_id="session-1",
        user_id="user-1",
        max_tokens=512,
    )

    assert request.to_command() == RecallContextCommand(
        query="Where does the user work?",
        session_id="session-1",
        user_id="user-1",
        max_tokens=512,
    )


def test_search_request_builds_search_command():
    request = SearchRequest(query="favorite color", user_id="user-1", limit=5)

    assert request.to_command() == SearchMemoriesCommand(
        query="favorite color",
        user_id="user-1",
        session_id=None,
        limit=5,
    )


def test_format_user_memories_preserves_public_contract():
    memory = SimpleNamespace(
        id="memory-1",
        type="fact",
        key="location",
        value="Lives in Berlin",
        confidence=0.95,
        session_id="session-1",
        source_turn_id="turn-1",
        created_at=datetime(2025, 5, 1, 10, 30, 0),
        updated_at=datetime(2025, 5, 2, 11, 45, 0),
        supersedes="old-memory",
        superseded_by=None,
        active=True,
    )

    response = format_user_memories([memory])

    assert response.model_dump() == {
        "memories": [
            {
                "id": "memory-1",
                "type": "fact",
                "key": "location",
                "value": "Lives in Berlin",
                "confidence": 0.95,
                "source_session": "session-1",
                "source_turn": "turn-1",
                "created_at": "2025-05-01T10:30:00",
                "updated_at": "2025-05-02T11:45:00",
                "supersedes": "old-memory",
                "superseded_by": None,
                "active": True,
            }
        ]
    }


def test_format_timestamp_allows_missing_timestamps():
    assert format_timestamp(None) is None
