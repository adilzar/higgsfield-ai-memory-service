from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from pydantic import BaseModel, Field

from src.ingestion.intake import IngestTurnCommand, TurnMessage
from src.storage.models import Memory
from src.recall import RecallContextCommand
from src.core.search import SearchMemoriesCommand


class Message(BaseModel):
    role: str
    content: str
    name: str | None = None


class TurnRequest(BaseModel):
    session_id: str
    user_id: str | None = None
    messages: list[Message]
    timestamp: str | None = None
    metadata: dict | None = None

    def to_command(self) -> IngestTurnCommand:
        return IngestTurnCommand(
            session_id=self.session_id,
            user_id=self.user_id,
            messages=[
                TurnMessage(message.role, message.content, message.name)
                for message in self.messages
            ],
            timestamp=self.timestamp,
            metadata=self.metadata,
        )


class RecallRequest(BaseModel):
    query: str
    session_id: str
    user_id: str | None = None
    max_tokens: int = Field(default=1024, gt=0)

    def to_command(self) -> RecallContextCommand:
        return RecallContextCommand(
            query=self.query,
            session_id=self.session_id,
            user_id=self.user_id,
            max_tokens=self.max_tokens,
        )


class SearchRequest(BaseModel):
    query: str
    session_id: str | None = None
    user_id: str | None = None
    limit: int = Field(default=10, gt=0)

    def to_command(self) -> SearchMemoriesCommand:
        return SearchMemoriesCommand(
            query=self.query,
            user_id=self.user_id,
            session_id=self.session_id,
            limit=self.limit,
        )


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
