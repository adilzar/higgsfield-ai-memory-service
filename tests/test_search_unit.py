from datetime import datetime, timezone

from src.search import format_search_result


def test_format_search_result_preserves_contract_shape():
    created_at = datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc)

    result = format_search_result({
        "value": "Favorite color is blue",
        "rrf_score": 1 / 61,
        "session_id": "session-1",
        "created_at": created_at,
        "type": "preference",
        "key": "favorite_color",
    })

    assert result == {
        "content": "Favorite color is blue",
        "score": 0.0164,
        "session_id": "session-1",
        "timestamp": created_at.isoformat(),
        "metadata": {"type": "preference", "key": "favorite_color"},
    }
