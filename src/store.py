from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Memory, Turn

RRF_K = 60


async def fetch_active_memory_models(
    db: AsyncSession, user_id: str | None, session_id: str
) -> list[Memory]:
    if user_id:
        predicate = Memory.user_id == user_id
    else:
        predicate = Memory.session_id == session_id

    result = await db.execute(
        select(Memory).where(predicate, Memory.active == True).order_by(Memory.created_at.desc())
    )
    return list(result.scalars().all())


async def vector_search_memories(
    db: AsyncSession,
    query_embedding: list[float],
    user_id: str | None,
    session_id: str | None,
    limit: int,
) -> list[dict]:
    conditions = ["active = true"]
    params: dict = {"embedding": str(query_embedding), "limit": limit}

    if user_id:
        conditions.append("user_id = :user_id")
        params["user_id"] = user_id
    if session_id:
        conditions.append("session_id = :session_id")
        params["session_id"] = session_id

    where = " AND ".join(conditions)
    sql = sa_text(
        f"""
        SELECT id, type, key, value, confidence, session_id, source_turn_id,
               created_at, updated_at, active, supersedes, superseded_by,
               1 - (embedding <=> CAST(:embedding AS vector)) as score
        FROM memories
        WHERE {where}
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """
    )
    result = await db.execute(sql, params)
    return [dict(r) for r in result.mappings().all()]


async def hybrid_search_memories(
    db: AsyncSession,
    query: str,
    query_embedding: list[float],
    user_id: str | None,
    session_id: str,
    limit: int,
) -> list[dict]:
    where, scope_params = _recall_memory_scope(user_id, session_id)
    return await _hybrid_search_memory_rows(db, query, query_embedding, where, scope_params, limit)


async def hybrid_search_filtered_memories(
    db: AsyncSession,
    query: str,
    query_embedding: list[float],
    user_id: str | None,
    session_id: str | None,
    limit: int,
) -> list[dict]:
    where, scope_params = _filtered_memory_scope(user_id, session_id)
    return await _hybrid_search_memory_rows(db, query, query_embedding, where, scope_params, limit)


async def _hybrid_search_memory_rows(
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

    vector_sql = sa_text(
        f"""
        SELECT id, type, key, value, confidence, session_id, source_turn_id, created_at, updated_at,
               active, supersedes, superseded_by,
               1 - (embedding <=> CAST(:embedding AS vector)) as vec_score
        FROM memories
        WHERE {where}
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """
    )
    vec_result = await db.execute(vector_sql, params)
    vec_rows = [dict(row) for row in vec_result.mappings().all()]

    fts_sql = sa_text(
        f"""
        SELECT id, type, key, value, confidence, session_id, source_turn_id, created_at, updated_at,
               active, supersedes, superseded_by,
               ts_rank(to_tsvector('english', value), plainto_tsquery('english', :query)) as fts_score
        FROM memories
        WHERE {where}
          AND to_tsvector('english', value) @@ plainto_tsquery('english', :query)
        ORDER BY fts_score DESC
        LIMIT :limit
    """
    )
    fts_result = await db.execute(fts_sql, params)
    fts_rows = [dict(row) for row in fts_result.mappings().all()]

    return fuse_ranked_memory_rows(vec_rows, fts_rows)


def _recall_memory_scope(user_id: str | None, session_id: str) -> tuple[str, dict]:
    if user_id:
        return "active = true AND user_id = :user_id", {"user_id": user_id}
    return "active = true AND session_id = :session_id", {"session_id": session_id}


def _filtered_memory_scope(user_id: str | None, session_id: str | None) -> tuple[str, dict]:
    conditions = ["active = true"]
    params = {}

    if user_id:
        conditions.append("user_id = :user_id")
        params["user_id"] = user_id
    if session_id:
        conditions.append("session_id = :session_id")
        params["session_id"] = session_id

    return " AND ".join(conditions), params


async def fetch_scope_memories(
    db: AsyncSession,
    user_id: str | None,
    session_id: str,
    include_inactive: bool = False,
) -> list[dict]:
    active_clause = "" if include_inactive else "AND active = true"
    sql = sa_text(
        f"""
        SELECT id, type, key, value, confidence, session_id, source_turn_id, created_at, updated_at,
               active, supersedes, superseded_by
        FROM memories
        WHERE (user_id = :user_id OR (:user_id IS NULL AND session_id = :session_id))
          {active_clause}
        ORDER BY active DESC, created_at DESC
    """
    )
    result = await db.execute(sql, {"user_id": user_id, "session_id": session_id})
    return [dict(r) for r in result.mappings().all()]


async def fetch_recent_turns(db: AsyncSession, session_id: str, limit: int = 5) -> list[dict]:
    sql = sa_text(
        """
        SELECT id, content_text, timestamp
        FROM turns
        WHERE session_id = :session_id
        ORDER BY timestamp DESC
        LIMIT :limit
    """
    )
    result = await db.execute(sql, {"session_id": session_id, "limit": limit})
    return [dict(r) for r in result.mappings().all()]


async def fetch_user_memory_models(db: AsyncSession, user_id: str) -> list[Memory]:
    result = await db.execute(
        select(Memory).where(Memory.user_id == user_id).order_by(Memory.created_at.desc())
    )
    return list(result.scalars().all())


async def delete_session_data(db: AsyncSession, session_id: str) -> None:
    # Reactivate memories that were superseded by memories in this session
    await db.execute(
        sa_text(
            """
            UPDATE memories SET active = true, superseded_by = NULL
            WHERE id IN (
                SELECT supersedes FROM memories
                WHERE session_id = :session_id AND supersedes IS NOT NULL
            )
        """
        ),
        {"session_id": session_id},
    )
    await db.execute(delete(Memory).where(Memory.session_id == session_id))
    await db.execute(delete(Turn).where(Turn.session_id == session_id))
    await db.commit()


async def delete_user_data(db: AsyncSession, user_id: str) -> None:
    await db.execute(delete(Memory).where(Memory.user_id == user_id))
    await db.execute(delete(Turn).where(Turn.user_id == user_id))
    await db.commit()


def fuse_ranked_memory_rows(vec_rows: list[dict], fts_rows: list[dict]) -> list[dict]:
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
