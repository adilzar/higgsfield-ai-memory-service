from types import SimpleNamespace

import pytest

from src.intake import (
    IngestTurnCommand,
    TurnMessage,
    format_turn_messages,
    memory_refs,
    message_to_dict,
    parse_turn_timestamp,
    persist_extracted_memories,
)


def test_format_turn_messages_preserves_roles_and_order():
    messages = [
        TurnMessage(role="user", content="I work at Stripe."),
        TurnMessage(role="assistant", content="How long have you been there?"),
        TurnMessage(role="tool", content='{"company":"Stripe"}', name="lookup"),
    ]

    assert format_turn_messages(messages) == (
        "user: I work at Stripe.\n"
        "assistant: How long have you been there?\n"
        'tool: {"company":"Stripe"}'
    )


def test_message_to_dict_preserves_name_key():
    assert message_to_dict(TurnMessage("tool", "{}", "lookup")) == {
        "role": "tool",
        "content": "{}",
        "name": "lookup",
    }


def test_parse_turn_timestamp_accepts_z_suffix():
    parsed = parse_turn_timestamp("2025-03-15T10:30:00Z")

    assert parsed.isoformat() == "2025-03-15T10:30:00+00:00"


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

    monkeypatch.setattr("src.intake.embed_texts", lambda values: [[0.0] * 384 for _ in values])

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
    cmd = IngestTurnCommand(
        session_id="session-1",
        user_id="user-1",
        messages=[TurnMessage(role="user", content="I just started at Notion.")],
    )

    await persist_extracted_memories(
        db,
        cmd,
        "turn-1",
        [old],
        [
            {
                "type": "fact",
                "key": "employment",
                "value": "Works at Notion as a PM",
                "confidence": 0.95,
                "supersedes_key": "employment",
            }
        ],
    )

    assert old.active is False
    assert old.updated_at is not None
    assert len(db.added) == 1
    new_memory = db.added[0]
    assert old.superseded_by == new_memory.id
    assert new_memory.supersedes == "old-employment"
    assert new_memory.value == "Works at Notion as a PM"
