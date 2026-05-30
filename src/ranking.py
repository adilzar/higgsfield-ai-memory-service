"""Pure ranking algorithms — no database dependency."""

from __future__ import annotations

RRF_K = 60


def fuse_ranked_memory_rows(vec_rows: list[dict], fts_rows: list[dict]) -> list[dict]:
    """Reciprocal Rank Fusion: combine vector and FTS result sets."""
    scores: dict[str, float] = {}
    memory_map: dict[str, dict] = {}

    for rank, row in enumerate(vec_rows):
        mid = row["id"]
        scores[mid] = scores.get(mid, 0) + 1.0 / (RRF_K + rank + 1)
        memory_map[mid] = dict(row)

    for rank, row in enumerate(fts_rows):
        mid = row["id"]
        scores[mid] = scores.get(mid, 0) + 1.0 / (RRF_K + rank + 1)
        if mid not in memory_map:
            memory_map[mid] = dict(row)
        else:
            memory_map[mid]["fts_score"] = row.get("fts_score")

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for mid, score in ranked:
        entry = memory_map[mid]
        entry["rrf_score"] = score
        results.append(entry)

    return results
