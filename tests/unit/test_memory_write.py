from types import SimpleNamespace

import pytest

from src.ingestion.extraction import ExtractedMemory
from src.ingestion.memory_write import MemoryWriteContext, memory_refs, persist_extracted_memories


def test_memory_refs_exposes_extraction_context():
    old = SimpleNamespace(
        id="mem-1",
        key="employment",
        type="fact",
        value="Works at Stripe as an engineer",
    )

    assert memory_refs([old]) == [
        {
            "id": "mem-1",
            "key": "employment",
            "type": "fact",
            "value": "Works at Stripe as an engineer",
        }
    ]


@pytest.mark.asyncio
async def test_persist_extracted_memories_marks_superseded(monkeypatch):
    class FakeDb:
        def __init__(self):
            self.added = []

        def add(self, obj):
            self.added.append(obj)

    monkeypatch.setattr(
        "src.memory_write.embed_texts", lambda values: [[0.0] * 384 for _ in values]
    )

    db = FakeDb()
    old = SimpleNamespace(
        id="old-employment",
        key="employment",
        type="fact",
        value="Works at Stripe as an engineer",
        active=True,
        superseded_by=None,
        updated_at=None,
    )

    await persist_extracted_memories(
        db,
        MemoryWriteContext(
            session_id="session-1",
            user_id="user-1",
            source_turn_id="turn-1",
        ),
        [old],
        [
            ExtractedMemory(
                type="fact",
                key="employment",
                value="Works at Notion as a PM",
                confidence=0.95,
                supersedes_key="employment",
            )
        ],
    )

    assert old.active is False
    assert old.updated_at is not None
    assert len(db.added) == 1
    new_memory = db.added[0]
    assert old.superseded_by == new_memory.id
    assert new_memory.supersedes == "old-employment"
    assert new_memory.value == "Works at Notion as a PM"
