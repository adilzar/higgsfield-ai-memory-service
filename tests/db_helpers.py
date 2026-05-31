from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.database import async_session, init_db
from src.storage.models import Memory, Turn


@dataclass(frozen=True)
class DbScenario:
    prefix: str
    unique: str = field(default_factory=lambda: uuid4().hex)

    @property
    def user_id(self) -> str:
        return f"{self.prefix}-user-{self.unique}"

    def session_id(self, label: str) -> str:
        return f"{self.prefix}-session-{label}-{self.unique}"

    def turn_id(self, label: str) -> str:
        return f"{self.prefix}-turn-{label}-{self.unique}"

    def memory_id(self, label: str) -> str:
        return f"{self.prefix}-memory-{label}-{self.unique}"

    def add_turn(self, db: AsyncSession, label: str) -> str:
        turn_id = self.turn_id(label)
        db.add(
            Turn(
                id=turn_id,
                session_id=self.session_id(label),
                user_id=self.user_id,
                messages=[],
                timestamp=datetime.utcnow(),
                content_text="test turn",
            )
        )
        return turn_id

    def add_memory(
        self,
        db: AsyncSession,
        label: str,
        *,
        value: str,
        active: bool,
        supersedes: str | None = None,
        superseded_by: str | None = None,
        key: str = "location",
        memory_type: str = "fact",
    ) -> str:
        memory_id = self.memory_id(label)
        db.add(
            Memory(
                id=memory_id,
                user_id=self.user_id,
                session_id=self.session_id(label),
                source_turn_id=self.turn_id(label),
                type=memory_type,
                key=key,
                value=value,
                active=active,
                supersedes=supersedes,
                superseded_by=superseded_by,
            )
        )
        return memory_id

    async def cleanup(self, db: AsyncSession) -> None:
        await db.execute(delete(Memory).where(Memory.user_id == self.user_id))
        await db.execute(delete(Turn).where(Turn.user_id == self.user_id))
        await db.commit()


@asynccontextmanager
async def db_scenario(prefix: str):
    await init_db()
    scenario = DbScenario(prefix)
    async with async_session() as db:
        try:
            yield db, scenario
        finally:
            await scenario.cleanup(db)
