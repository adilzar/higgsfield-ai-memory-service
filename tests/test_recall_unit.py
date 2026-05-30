from src.recall.budget import assemble_context
from src.recall.selection import _select_recall_memories


def memory(
    id_: str,
    key: str,
    value: str,
    *,
    type_: str = "fact",
    active: bool = True,
    score: float = 0.1,
) -> dict:
    return {
        "id": id_,
        "type": type_,
        "key": key,
        "value": value,
        "confidence": 1.0,
        "session_id": "session-1",
        "source_turn_id": f"turn-{id_}",
        "created_at": "2025-03-01T10:00:00",
        "updated_at": "2025-03-01T10:00:00",
        "active": active,
        "supersedes": None,
        "superseded_by": None,
        "rrf_score": score,
    }


def test_noise_gate_rejects_unknown_specific_query():
    employment = memory("1", "employment", "Works at Notion as a PM")

    selected = _select_recall_memories(
        "What car does the user drive?",
        [employment],
        [employment],
    )

    assert selected == []


def test_noise_gate_rejects_unknown_favorite_query():
    opinion = memory(
        "1",
        "typescript_opinion",
        "TypeScript is fine for big projects but Python is better for scripts",
        type_="opinion",
    )

    selected = _select_recall_memories(
        "What is the user's favorite color?",
        [opinion],
        [opinion],
    )

    assert selected == []


def test_anchor_expansion_links_pet_and_location():
    pet = memory("1", "pet", "Has a dog named Biscuit")
    location = memory("2", "location", "Lives in Berlin")
    employment = memory("3", "employment", "Works at Notion as a PM")

    selected = _select_recall_memories(
        "What city does the user with the dog named Biscuit live in?",
        [pet],
        [pet, location, employment],
    )

    values = [m["value"] for m in selected]
    assert "Has a dog named Biscuit" in values
    assert "Lives in Berlin" in values
    assert "Works at Notion as a PM" not in values


def test_history_query_can_surface_inactive_previous_memory():
    current = memory("1", "employment", "Works at Notion as a PM")
    previous = memory("2", "employment", "Worked at Stripe as an engineer", active=False)

    selected = _select_recall_memories(
        "Tell me about the user's career history",
        [current],
        [current, previous],
    )
    context, _ = assemble_context(selected, [], 512)

    assert "Works at Notion as a PM" in context
    assert "Previously: Worked at Stripe as an engineer" in context
