from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.embeddings import embed_text
from src.recall.retrieval import hybrid_search_filtered_memories
from src.storage.rows import MemoryRow


@dataclass(frozen=True)
class SearchMemoriesCommand:
    query: str
    user_id: str | None
    session_id: str | None
    limit: int


async def search_memories(db: AsyncSession, cmd: SearchMemoriesCommand) -> list[dict]:
    query_embedding = embed_text(cmd.query)
    rows = await hybrid_search_filtered_memories(
        db,
        cmd.query,
        query_embedding,
        cmd.user_id,
        cmd.session_id,
        cmd.limit,
    )
    return [format_search_result(row) for row in rows]


def format_search_result(row: MemoryRow) -> dict:
    return {
        "content": row.value,
        "score": round(float(row.rrf_score), 4),
        "session_id": row.session_id,
        "timestamp": row.created_at.isoformat() if row.created_at else None,
        "metadata": {"type": row.type, "key": row.key},
    }
