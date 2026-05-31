"""Pure ranking algorithms — no database dependency."""

from __future__ import annotations

from src.storage.rows import MemoryRow

RRF_K = 60


def fuse_ranked_memory_rows(
    vec_rows: list[MemoryRow], fts_rows: list[MemoryRow]
) -> list[MemoryRow]:
    """Reciprocal Rank Fusion: combine vector and FTS result sets."""
    scores: dict[str, float] = {}
    memory_map: dict[str, MemoryRow] = {}

    for rank, row in enumerate(vec_rows):
        mid = row.id
        scores[mid] = scores.get(mid, 0) + 1.0 / (RRF_K + rank + 1)
        memory_map[mid] = row

    for rank, row in enumerate(fts_rows):
        mid = row.id
        scores[mid] = scores.get(mid, 0) + 1.0 / (RRF_K + rank + 1)
        if mid not in memory_map:
            memory_map[mid] = row
        else:
            memory_map[mid] = memory_map[mid].with_scores(fts_score=row.fts_score)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for mid, score in ranked:
        results.append(memory_map[mid].with_scores(rrf_score=score))

    return results
