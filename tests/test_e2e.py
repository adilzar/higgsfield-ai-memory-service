"""
End-to-end tests for the Memory Service.
Tests multi-turn scenarios, fact evolution, cross-session recall,
token budget behavior, and supersession chain integrity.

Requires the service running at http://localhost:8080
"""

import httpx
import pytest

BASE_URL = "http://localhost:8080"


@pytest.fixture
def client():
    return httpx.Client(base_url=BASE_URL, timeout=60.0)


@pytest.fixture(autouse=True)
def cleanup(client):
    yield
    for uid in (
        "e2e-user-1",
        "e2e-user-2",
        "e2e-evolution",
        "e2e-budget",
        "e2e-multihop",
    ):
        client.delete(f"/users/{uid}")


def ingest(
    client,
    session_id,
    user_id,
    user_msg,
    asst_msg="Got it.",
    timestamp="2025-03-01T10:00:00Z",
):
    r = client.post(
        "/turns",
        json={
            "session_id": session_id,
            "user_id": user_id,
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": asst_msg},
            ],
            "timestamp": timestamp,
            "metadata": {},
        },
    )
    assert r.status_code == 201
    return r.json()["id"]


def recall(client, query, session_id, user_id, max_tokens=1024):
    r = client.post(
        "/recall",
        json={
            "query": query,
            "session_id": session_id,
            "user_id": user_id,
            "max_tokens": max_tokens,
        },
    )
    assert r.status_code == 200
    return r.json()


class TestFactEvolution:
    """Test that contradictions are detected and old facts superseded."""

    def test_employment_update(self, client):
        ingest(
            client,
            "s1",
            "e2e-evolution",
            "I work at Stripe as a backend engineer.",
            timestamp="2025-01-01T10:00:00Z",
        )
        ingest(
            client,
            "s2",
            "e2e-evolution",
            "I just joined Notion as a product manager. Left Stripe last week.",
            timestamp="2025-03-01T10:00:00Z",
        )

        # Recall should return current fact
        result = recall(client, "Where does the user work?", "s3", "e2e-evolution")
        ctx = result["context"].lower()
        assert "notion" in ctx
        # Should NOT present Stripe as current
        assert "stripe" not in ctx or "previously" in ctx or "left" in ctx

    def test_location_update(self, client):
        ingest(
            client,
            "s1",
            "e2e-evolution",
            "I live in San Francisco.",
            timestamp="2025-01-01T10:00:00Z",
        )
        ingest(
            client,
            "s2",
            "e2e-evolution",
            "I just moved to Berlin last month.",
            timestamp="2025-03-01T10:00:00Z",
        )

        result = recall(client, "Where does the user live?", "s3", "e2e-evolution")
        ctx = result["context"].lower()
        assert "berlin" in ctx

    def test_supersession_chain_visible_in_memories(self, client):
        ingest(
            client,
            "s1",
            "e2e-evolution",
            "I work at Stripe.",
            timestamp="2025-01-01T10:00:00Z",
        )
        ingest(
            client,
            "s2",
            "e2e-evolution",
            "I now work at Notion.",
            timestamp="2025-03-01T10:00:00Z",
        )

        r = client.get("/users/e2e-evolution/memories")
        memories = r.json()["memories"]

        active = [m for m in memories if m["active"]]
        inactive = [m for m in memories if not m["active"]]

        # Should have at least one active (Notion) and one inactive (Stripe)
        assert any("notion" in m["value"].lower() for m in active)
        assert any("stripe" in m["value"].lower() for m in inactive)

        # Supersession pointers should be set
        notion_mem = next(m for m in active if "notion" in m["value"].lower())
        assert notion_mem["supersedes"] is not None

    def test_correction_handling(self, client):
        ingest(
            client,
            "s1",
            "e2e-evolution",
            "I have two kids.",
            timestamp="2025-01-01T10:00:00Z",
        )
        ingest(
            client,
            "s2",
            "e2e-evolution",
            "Actually, sorry — I have three kids, not two.",
            timestamp="2025-03-01T10:00:00Z",
        )

        result = recall(client, "How many kids does the user have?", "s3", "e2e-evolution")
        ctx = result["context"].lower()
        assert "three" in ctx


class TestCrossSessionKnowledge:
    """Test that memories extracted in one session are available in another."""

    def test_facts_available_across_sessions(self, client):
        ingest(
            client,
            "session-A",
            "e2e-user-1",
            "I'm a vegetarian and allergic to shellfish.",
        )
        ingest(
            client,
            "session-B",
            "e2e-user-1",
            "I have a golden retriever named Biscuit.",
        )

        # Query from a completely new session should find both
        result = recall(client, "What are the user's dietary needs?", "session-C", "e2e-user-1")
        ctx = result["context"].lower()
        assert "vegetarian" in ctx or "shellfish" in ctx

    def test_different_users_isolated(self, client):
        ingest(client, "s1", "e2e-user-1", "I'm a pilot in Dubai.")
        ingest(client, "s2", "e2e-user-2", "I'm a teacher in Oslo.")

        r1 = recall(client, "What does the user do?", "s3", "e2e-user-1")
        r2 = recall(client, "What does the user do?", "s4", "e2e-user-2")

        assert "pilot" in r1["context"].lower()
        assert "teacher" in r2["context"].lower()
        assert "teacher" not in r1["context"].lower()
        assert "pilot" not in r2["context"].lower()


class TestTokenBudget:
    """Test that recall respects max_tokens and prioritizes correctly."""

    def test_tight_budget_includes_facts(self, client):
        ingest(client, "s1", "e2e-budget", "I work at SpaceX as a rocket engineer.")
        ingest(
            client,
            "s1",
            "e2e-budget",
            "Yesterday I was debugging a memory leak in the telemetry service. "
            "It took 4 hours and involved tracing through the gRPC streaming layer.",
            timestamp="2025-03-02T10:00:00Z",
        )

        # Very tight budget — should prioritize the stable fact over the event
        result = recall(client, "Tell me about this user", "s2", "e2e-budget", max_tokens=64)
        ctx = result["context"].lower()
        # With only 64 tokens, the fact should be there
        assert "spacex" in ctx or "rocket" in ctx

    def test_large_budget_includes_more(self, client):
        ingest(client, "s1", "e2e-budget", "I work at SpaceX as a rocket engineer.")
        ingest(
            client,
            "s1",
            "e2e-budget",
            "I prefer concise answers. No fluff.",
            timestamp="2025-03-02T10:00:00Z",
        )

        result = recall(client, "Tell me about this user", "s2", "e2e-budget", max_tokens=1024)
        ctx = result["context"].lower()
        assert "spacex" in ctx or "rocket" in ctx
        assert "concise" in ctx or "fluff" in ctx


class TestMultiHop:
    """Test queries that require connecting multiple memories."""

    def test_connect_pet_and_location(self, client):
        ingest(client, "s1", "e2e-multihop", "I have a dog named Biscuit.")
        ingest(
            client,
            "s2",
            "e2e-multihop",
            "I live in Berlin.",
            timestamp="2025-03-02T10:00:00Z",
        )

        result = recall(
            client,
            "What city does the user with the dog named Biscuit live in?",
            "s3",
            "e2e-multihop",
        )
        ctx = result["context"].lower()
        # Both facts should be surfaced
        assert "berlin" in ctx
        assert "biscuit" in ctx


class TestSessionDelete:
    """Test that session deletion properly handles supersession chains."""

    def test_delete_superseding_session_reactivates_old_fact(self, client):
        ingest(
            client,
            "s1",
            "e2e-user-1",
            "I live in NYC.",
            timestamp="2025-01-01T10:00:00Z",
        )
        ingest(
            client,
            "s2",
            "e2e-user-1",
            "I just moved to Berlin.",
            timestamp="2025-03-01T10:00:00Z",
        )

        # Delete the session with the newer fact
        client.delete("/sessions/s2")

        # Old fact should be reactivated
        r = client.get("/users/e2e-user-1/memories")
        memories = r.json()["memories"]
        active = [m for m in memories if m["active"]]
        assert any(is_nyc_memory(m["value"]) for m in active)

        # Recall should return the old fact
        result = recall(client, "Where does the user live?", "s3", "e2e-user-1")
        assert is_nyc_memory(result["context"])


def is_nyc_memory(value: str) -> bool:
    normalized = value.lower()
    return "nyc" in normalized or "new york" in normalized


class TestImplicitFacts:
    """Test extraction of implicit/inferred facts."""

    def test_implicit_pet_from_activity(self, client):
        ingest(
            client,
            "s1",
            "e2e-user-1",
            "I was walking Biscuit along the river this morning. She loved chasing the ducks.",
        )

        r = client.get("/users/e2e-user-1/memories")
        memories = r.json()["memories"]
        values = " ".join(m["value"].lower() for m in memories)
        # Should extract that user has a pet named Biscuit
        assert "biscuit" in values

    def test_implicit_relationship(self, client):
        ingest(
            client,
            "s1",
            "e2e-user-1",
            "I need to pick up my daughter from school at 3pm today.",
        )

        r = client.get("/users/e2e-user-1/memories")
        memories = r.json()["memories"]
        values = " ".join(m["value"].lower() for m in memories)
        assert "daughter" in values


class TestMultiMessageTurns:
    """Test handling of turns with tool calls and multiple messages."""

    def test_tool_call_turn(self, client):
        r = client.post(
            "/turns",
            json={
                "session_id": "s1",
                "user_id": "e2e-user-1",
                "messages": [
                    {"role": "user", "content": "What's the weather in my city?"},
                    {
                        "role": "tool",
                        "name": "get_weather",
                        "content": "Berlin: 15°C, partly cloudy",
                    },
                    {
                        "role": "assistant",
                        "content": "It's 15°C and partly cloudy in Berlin right now.",
                    },
                ],
                "timestamp": "2025-03-01T10:00:00Z",
                "metadata": {},
            },
        )
        assert r.status_code == 201

    def test_multi_message_extraction(self, client):
        r = client.post(
            "/turns",
            json={
                "session_id": "s1",
                "user_id": "e2e-user-1",
                "messages": [
                    {
                        "role": "user",
                        "content": "I'm preparing for a system design interview at Google.",
                    },
                    {
                        "role": "assistant",
                        "content": "I can help with that. What topics are you focusing on?",
                    },
                    {
                        "role": "user",
                        "content": "Distributed systems and database design mostly.",
                    },
                    {
                        "role": "assistant",
                        "content": "Great choices. Let's start with a classic: design a URL shortener.",
                    },
                ],
                "timestamp": "2025-03-01T10:00:00Z",
                "metadata": {},
            },
        )
        assert r.status_code == 201

        r = client.get("/users/e2e-user-1/memories")
        memories = r.json()["memories"]
        values = " ".join(m["value"].lower() for m in memories)
        assert "interview" in values or "google" in values
