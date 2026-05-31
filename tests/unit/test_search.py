from datetime import datetime, timezone

from src.core.search import format_search_result
from src.storage.rows import MemoryRow


def test_format_search_result_preserves_contract_shape():
    created_at = datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc)

    result = format_search_result(
        MemoryRow(
            id="memory-1",
            type="preference",
            key="favorite_color",
            value="Favorite color is blue",
            confidence=1.0,
            session_id="session-1",
            source_turn_id="turn-1",
            created_at=created_at,
            rrf_score=1 / 61,
        )
    )

    assert result == {
        "content": "Favorite color is blue",
        "score": 0.0164,
        "session_id": "session-1",
        "timestamp": created_at.isoformat(),
        "metadata": {"type": "preference", "key": "favorite_color"},
    }
