from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.embeddings import embed_text
from src.ingestion.extraction import ExtractionError, extract_memories
from src.ingestion.memory_write import MemoryWriteContext, memory_refs, persist_extracted_memories
from src.storage.models import Turn
from src.storage.store import fetch_active_memory_models

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnMessage:
    role: str
    content: str
    name: str | None = None


@dataclass(frozen=True)
class IngestTurnCommand:
    session_id: str
    user_id: str | None
    messages: list[TurnMessage]
    timestamp: str | None = None
    metadata: dict | None = None


async def ingest_turn(db: AsyncSession, cmd: IngestTurnCommand) -> str:
    """Persist a Turn, extract Memory, and apply fact evolution synchronously."""
    turn_id = str(uuid.uuid4())
    content_text = format_turn_messages(cmd.messages)
    timestamp = parse_turn_timestamp(cmd.timestamp)

    turn = Turn(
        id=turn_id,
        session_id=cmd.session_id,
        user_id=cmd.user_id,
        messages=[message_to_dict(m) for m in cmd.messages],
        timestamp=timestamp,
        metadata_=cmd.metadata,
        content_text=content_text,
        embedding=embed_text(content_text),
    )
    db.add(turn)
    await db.flush()

    existing = await fetch_active_memory_models(db, cmd.user_id, cmd.session_id)
    try:
        result = extract_memories(content_text, memory_refs(existing))
    except ExtractionError as e:
        logger.warning("Extraction failed for turn %s: %s", turn_id, e)
        await db.commit()
        return turn_id

    await persist_extracted_memories(
        db,
        MemoryWriteContext(
            session_id=cmd.session_id,
            user_id=cmd.user_id,
            source_turn_id=turn_id,
        ),
        existing,
        result.memories,
    )

    await db.commit()
    return turn_id


def format_turn_messages(messages: list[TurnMessage]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


def parse_turn_timestamp(timestamp: str | None) -> datetime:
    if timestamp:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return datetime.utcnow()


def message_to_dict(message: TurnMessage) -> dict:
    return {"role": message.role, "content": message.content, "name": message.name}
