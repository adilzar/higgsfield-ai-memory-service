from src.recall.ranking import fuse_ranked_memory_rows
from src.recall.retrieval import filtered_memory_scope, recall_memory_scope
from src.storage.rows import MemoryRow


def memory_row(
    id_: str,
    value: str,
    *,
    vec_score: float | None = None,
    fts_score: float | None = None,
) -> MemoryRow:
    return MemoryRow(
        id=id_,
        type="fact",
        key=id_,
        value=value,
        confidence=1.0,
        session_id="session-1",
        source_turn_id=f"turn-{id_}",
        vec_score=vec_score,
        fts_score=fts_score,
    )


def test_fuse_ranked_memory_rows_merges_vector_and_fts_scores():
    vec_rows = [
        memory_row("a", "Biscuit is a dog", vec_score=0.7),
        memory_row("b", "Lives in Berlin", vec_score=0.6),
    ]
    fts_rows = [
        memory_row("b", "Lives in Berlin", fts_score=0.8),
        memory_row("c", "Vegetarian", fts_score=0.5),
    ]

    fused = fuse_ranked_memory_rows(vec_rows, fts_rows)

    assert [row.id for row in fused] == ["b", "a", "c"]
    assert fused[0].fts_score == 0.8
    assert fused[0].rrf_score > fused[1].rrf_score


def test_fuse_ranked_memory_rows_preserves_single_signal_rows():
    fused = fuse_ranked_memory_rows(
        [memory_row("a", "Works at Notion")],
        [memory_row("b", "Has a dog named Biscuit", fts_score=0.9)],
    )

    by_id = {row.id: row for row in fused}
    assert by_id["a"].value == "Works at Notion"
    assert by_id["b"].fts_score == 0.9
    assert all(row.rrf_score > 0 for row in fused)


def test_recall_memory_scope_uses_user_history_when_user_id_exists():
    where, params = recall_memory_scope("user-1", "session-1")

    assert where == "active = true AND user_id = :user_id"
    assert params == {"user_id": "user-1"}


def test_filtered_memory_scope_intersects_user_and_session_filters():
    where, params = filtered_memory_scope("user-1", "session-1")

    assert where == "active = true AND user_id = :user_id AND session_id = :session_id"
    assert params == {"user_id": "user-1", "session_id": "session-1"}


def test_filtered_memory_scope_allows_global_search_when_unscoped():
    where, params = filtered_memory_scope(None, None)

    assert where == "active = true"
    assert params == {}
