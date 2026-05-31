from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.embeddings import embed_text
from src.recall.budget import RecallContext, assemble_context
from src.recall.planning import build_recall_plan, needs_history
from src.recall.retrieval import hybrid_search_memories
from src.storage.rows import MemoryRow
from src.storage.store import fetch_recent_turns, fetch_scope_memories


@dataclass(frozen=True)
class RecallContextCommand:
    query: str
    session_id: str
    user_id: str | None
    max_tokens: int


async def hybrid_retrieve(
    db: AsyncSession, query: str, user_id: str | None, session_id: str, limit: int = 20
) -> list[MemoryRow]:
    """Hybrid retrieval: vector similarity + full-text search, fused with RRF."""
    query_embedding = embed_text(query)
    return await hybrid_search_memories(db, query, query_embedding, user_id, session_id, limit)


async def build_recall_context(
    db: AsyncSession,
    cmd: RecallContextCommand,
) -> RecallContext:
    """Build prompt Context and Citations for /recall."""
    include_inactive = needs_history(cmd.query)
    retrieved = await hybrid_retrieve(db, cmd.query, cmd.user_id, cmd.session_id)
    scope_memories = await fetch_scope_memories(db, cmd.user_id, cmd.session_id, include_inactive)
    plan = build_recall_plan(cmd.query, retrieved, scope_memories)

    if not plan.memories:
        return RecallContext.empty()

    recent = await fetch_recent_turns(db, cmd.session_id)
    return assemble_context(plan.memories, recent, cmd.max_tokens)
