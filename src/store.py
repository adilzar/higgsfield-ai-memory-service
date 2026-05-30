"""Data access layer — fetches, deletes, and simple queries."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Memory, Turn


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
