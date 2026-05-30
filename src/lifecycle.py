from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Memory, Turn


async def delete_session_data(db: AsyncSession, session_id: str) -> None:
    memories = await fetch_session_memories(db, session_id)
    if memories:
        await repair_supersession_for_deleted_memories(db, memories)

    await db.execute(delete(Memory).where(Memory.session_id == session_id))
    await db.execute(delete(Turn).where(Turn.session_id == session_id))
    await db.commit()


async def delete_user_data(db: AsyncSession, user_id: str) -> None:
    await db.execute(delete(Memory).where(Memory.user_id == user_id))
    await db.execute(delete(Turn).where(Turn.user_id == user_id))
    await db.commit()


async def fetch_session_memories(db: AsyncSession, session_id: str) -> list[Memory]:
    result = await db.execute(select(Memory).where(Memory.session_id == session_id))
    return list(result.scalars().all())


async def repair_supersession_for_deleted_memories(
    db: AsyncSession, deleted_memories: list[Memory]
) -> None:
    deleted_by_id = {memory.id: memory for memory in deleted_memories}
    external_children = await fetch_external_supersession_children(db, set(deleted_by_id))
    now = datetime.utcnow()
    stitched_ancestor_ids: set[str] = set()

    for child in external_children:
        removed_parent = deleted_by_id.get(child.supersedes)
        if not removed_parent:
            continue

        ancestor_id = nearest_external_ancestor_id(removed_parent, deleted_by_id)
        child.supersedes = ancestor_id
        child.updated_at = now

        if ancestor_id:
            ancestor = await db.get(Memory, ancestor_id)
            if ancestor:
                ancestor.active = False
                ancestor.superseded_by = child.id
                ancestor.updated_at = now
                stitched_ancestor_ids.add(ancestor.id)

    for memory in deleted_memories:
        if not memory.active:
            continue

        ancestor_id = nearest_external_ancestor_id(memory, deleted_by_id)
        if not ancestor_id or ancestor_id in stitched_ancestor_ids:
            continue

        ancestor = await db.get(Memory, ancestor_id)
        if ancestor:
            ancestor.active = True
            ancestor.superseded_by = None
            ancestor.updated_at = now


async def fetch_external_supersession_children(
    db: AsyncSession, deleted_ids: set[str]
) -> list[Memory]:
    result = await db.execute(
        select(Memory).where(Memory.supersedes.in_(deleted_ids), ~Memory.id.in_(deleted_ids))
    )
    return list(result.scalars().all())


def nearest_external_ancestor_id(memory: Memory, deleted_by_id: dict[str, Memory]) -> str | None:
    ancestor_id = memory.supersedes
    visited: set[str] = set()

    while ancestor_id in deleted_by_id and ancestor_id not in visited:
        visited.add(ancestor_id)
        ancestor_id = deleted_by_id[ancestor_id].supersedes

    return ancestor_id
