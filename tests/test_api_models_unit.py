from datetime import datetime
from types import SimpleNamespace

from src.api_models import format_timestamp, format_user_memories


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
