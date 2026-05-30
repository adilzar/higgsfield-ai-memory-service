from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from pydantic import BaseModel

from src.models import Memory


class UserMemoryResponse(BaseModel):
    id: str
    type: str
    key: str
    value: str
    confidence: float | None
    source_session: str
    source_turn: str
    created_at: str | None
    updated_at: str | None
    supersedes: str | None
    superseded_by: str | None
    active: bool | None


class UserMemoriesResponse(BaseModel):
    memories: list[UserMemoryResponse]


def format_user_memories(memories: Iterable[Memory]) -> UserMemoriesResponse:
    return UserMemoriesResponse(memories=[format_user_memory(memory) for memory in memories])


def format_user_memory(memory: Memory) -> UserMemoryResponse:
    return UserMemoryResponse(
        id=memory.id,
        type=memory.type,
        key=memory.key,
        value=memory.value,
        confidence=memory.confidence,
        source_session=memory.session_id,
        source_turn=memory.source_turn_id,
        created_at=format_timestamp(memory.created_at),
        updated_at=format_timestamp(memory.updated_at),
        supersedes=memory.supersedes,
        superseded_by=memory.superseded_by,
        active=memory.active,
    )


def format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()
