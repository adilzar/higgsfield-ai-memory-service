"""Hybrid retrieval: vector + FTS search with scope builders."""

from __future__ import annotations

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from src.recall.ranking import fuse_ranked_memory_rows


def recall_memory_scope(user_id: str | None, session_id: str) -> tuple[str, dict]:
    """Scope for /recall: all active memories for the user, or session-scoped if no user."""
    if user_id:
        return "active = true AND user_id = :user_id", {"user_id": user_id}
    return "active = true AND session_id = :session_id", {"session_id": session_id}


def filtered_memory_scope(user_id: str | None, session_id: str | None) -> tuple[str, dict]:
    """Scope for /search: intersect provided filters."""
    conditions = ["active = true"]
    params: dict = {}

    if user_id:
        conditions.append("user_id = :user_id")
        params["user_id"] = user_id
    if session_id:
        conditions.append("session_id = :session_id")
        params["session_id"] = session_id

    return " AND ".join(conditions), params


async def hybrid_search_memories(
    db: AsyncSession,
    query: str,
    query_embedding: list[float],
    user_id: str | None,
    session_id: str,
    limit: int,
) -> list[dict]:
    """Hybrid search scoped for /recall."""
    where, scope_params = recall_memory_scope(user_id, session_id)
    return await _hybrid_search(db, query, query_embedding, where, scope_params, limit)


async def hybrid_search_filtered_memories(
    db: AsyncSession,
    query: str,
    query_embedding: list[float],
    user_id: str | None,
    session_id: str | None,
    limit: int,
) -> list[dict]:
    """Hybrid search scoped for /search."""
    where, scope_params = filtered_memory_scope(user_id, session_id)
    return await _hybrid_search(db, query, query_embedding, where, scope_params, limit)


async def _hybrid_search(
    db: AsyncSession,
    query: str,
    query_embedding: list[float],
    where: str,
    scope_params: dict,
    limit: int,
) -> list[dict]:
    params = {
        **scope_params,
        "embedding": str(query_embedding),
        "query": query,
        "limit": limit,
    }

    vector_sql = sa_text(f"""
        SELECT id, type, key, value, confidence, session_id, source_turn_id, created_at, updated_at,
               active, supersedes, superseded_by,
               1 - (embedding <=> CAST(:embedding AS vector)) as vec_score
        FROM memories
        WHERE {where}
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)
    vec_result = await db.execute(vector_sql, params)
    vec_rows = [dict(row) for row in vec_result.mappings().all()]

    fts_sql = sa_text(f"""
        SELECT id, type, key, value, confidence, session_id, source_turn_id, created_at, updated_at,
               active, supersedes, superseded_by,
               ts_rank(to_tsvector('english', value), plainto_tsquery('english', :query)) as fts_score
        FROM memories
        WHERE {where}
          AND to_tsvector('english', value) @@ plainto_tsquery('english', :query)
        ORDER BY fts_score DESC
        LIMIT :limit
    """)
    fts_result = await db.execute(fts_sql, params)
    fts_rows = [dict(row) for row in fts_result.mappings().all()]

    return fuse_ranked_memory_rows(vec_rows, fts_rows)
