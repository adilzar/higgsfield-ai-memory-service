from src.store import fuse_ranked_memory_rows


def test_fuse_ranked_memory_rows_merges_vector_and_fts_scores():
    vec_rows = [
        {"id": "a", "value": "Biscuit is a dog", "vec_score": 0.7},
        {"id": "b", "value": "Lives in Berlin", "vec_score": 0.6},
    ]
    fts_rows = [
        {"id": "b", "value": "Lives in Berlin", "fts_score": 0.8},
        {"id": "c", "value": "Vegetarian", "fts_score": 0.5},
    ]

    fused = fuse_ranked_memory_rows(vec_rows, fts_rows)

    assert [row["id"] for row in fused] == ["b", "a", "c"]
    assert fused[0]["fts_score"] == 0.8
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_fuse_ranked_memory_rows_preserves_single_signal_rows():
    fused = fuse_ranked_memory_rows(
        [{"id": "a", "value": "Works at Notion"}],
        [{"id": "b", "value": "Has a dog named Biscuit", "fts_score": 0.9}],
    )

    by_id = {row["id"]: row for row in fused}
    assert by_id["a"]["value"] == "Works at Notion"
    assert by_id["b"]["fts_score"] == 0.9
    assert all("rrf_score" in row for row in fused)
