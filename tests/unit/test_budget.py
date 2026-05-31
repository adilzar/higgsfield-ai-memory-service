from src.recall.budget import Citation, assemble_context, estimate_tokens
from src.storage.rows import MemoryRow, RecentTurnRow


def memory(
    id_: str,
    type_: str,
    value: str,
    *,
    active: bool = True,
) -> MemoryRow:
    return MemoryRow(
        id=id_,
        type=type_,
        key=type_,
        value=value,
        confidence=1.0,
        session_id="session-1",
        source_turn_id=f"turn-{id_}",
        created_at="2025-03-01T10:00:00",
        updated_at="2025-03-01T10:00:00",
        active=active,
        rrf_score=0.1,
    )


def test_tight_budget_keeps_fact_before_recent_turn():
    memories = [memory("1", "fact", "Lives in Berlin")]
    recent_turns = [
        RecentTurnRow(
            id="recent-1",
            content_text="assistant: Long recent conversation that should be cut first",
            timestamp="2025-03-02T10:00:00",
        )
    ]

    context = assemble_context(memories, recent_turns, max_tokens=12)

    assert "Lives in Berlin" in context.text
    assert "Recent conversation context" not in context.text
    assert context.citations == (Citation(turn_id="turn-1", score=0.1, snippet="Lives in Berlin"),)


def test_budget_policy_caps_preferences_before_relevant_memories():
    memories = [
        memory("1", "fact", "Lives in Berlin"),
        memory("2", "preference", "Prefers concise answers"),
        memory(
            "3",
            "opinion",
            "TypeScript is fine for big projects but Python is better for small scripts",
        ),
    ]

    context = assemble_context(memories, [], max_tokens=24)

    assert "Lives in Berlin" in context.text
    assert "Prefers concise answers" in context.text
    assert "TypeScript is fine" not in context.text


def test_estimate_tokens_is_stable_for_budget_assertions():
    assert estimate_tokens("- Lives in Berlin") == 5
