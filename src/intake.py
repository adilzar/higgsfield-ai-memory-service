from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from src.embeddings import embed_text, embed_texts
from src.extraction import extract_memories
from src.models import Memory, Turn
from src.store import fetch_active_memory_models


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
    extracted = extract_memories(content_text, memory_refs(existing))
    await persist_extracted_memories(db, cmd, turn_id, existing, extracted)

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


def memory_refs(memories: list[Memory]) -> list[dict]:
    return [
        {"id": m.id, "key": m.key, "type": m.type, "value": m.value}
        for m in memories
    ]


async def persist_extracted_memories(
    db: AsyncSession,
    cmd: IngestTurnCommand,
    turn_id: str,
    existing: list[Memory],
    extracted: list[dict],
) -> None:
    if not extracted:
        return

    embeddings = embed_texts([m["value"] for m in extracted])
    existing_by_key = {m.key: m for m in reversed(existing)}

    for index, mem_data in enumerate(extracted):
        mem_id = str(uuid.uuid4())
        superseded = existing_by_key.get(mem_data.get("supersedes_key"))
        supersedes_id = None

        if superseded:
            supersedes_id = superseded.id
            superseded.active = False
            superseded.superseded_by = mem_id
            superseded.updated_at = datetime.utcnow()

        memory = Memory(
            id=mem_id,
            user_id=cmd.user_id,
            session_id=cmd.session_id,
            source_turn_id=turn_id,
            type=mem_data.get("type", "fact"),
            key=mem_data.get("key", "unknown"),
            value=mem_data["value"],
            confidence=mem_data.get("confidence", 1.0),
            active=True,
            supersedes=supersedes_id,
            embedding=embeddings[index],
        )
        db.add(memory)
