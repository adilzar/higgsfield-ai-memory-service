from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.embeddings import embed_texts
from src.ingestion.extraction import ExtractedMemory
from src.storage.models import Memory


@dataclass(frozen=True)
class MemoryWriteContext:
    session_id: str
    user_id: str | None
    source_turn_id: str


async def persist_extracted_memories(
    db: AsyncSession,
    context: MemoryWriteContext,
    existing: list[Memory],
    extracted: list[ExtractedMemory],
) -> None:
    if not extracted:
        return

    embeddings = embed_texts([m.value for m in extracted])
    existing_by_key = {m.key: m for m in reversed(existing)}

    for index, mem_data in enumerate(extracted):
        mem_id = str(uuid.uuid4())
        superseded = existing_by_key.get(mem_data.supersedes_key)
        supersedes_id = None

        if superseded:
            supersedes_id = superseded.id
            superseded.active = False
            superseded.superseded_by = mem_id
            superseded.updated_at = datetime.utcnow()

        memory = Memory(
            id=mem_id,
            user_id=context.user_id,
            session_id=context.session_id,
            source_turn_id=context.source_turn_id,
            type=mem_data.type,
            key=mem_data.key,
            value=mem_data.value,
            confidence=mem_data.confidence,
            active=True,
            supersedes=supersedes_id,
            embedding=embeddings[index],
        )
        db.add(memory)
