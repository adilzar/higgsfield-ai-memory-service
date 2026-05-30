"""
Integration tests for the Memory Service.
Run with: pytest tests/ -v
Requires the service running at http://localhost:8080
"""

import json
import time

import httpx
import pytest

BASE_URL = "http://localhost:8080"


@pytest.fixture
def client():
    return httpx.Client(base_url=BASE_URL, timeout=60.0)


@pytest.fixture(autouse=True)
def cleanup(client):
    yield
    # Cleanup test data
    client.delete("/users/test-contract-user")
    client.delete("/sessions/test-contract-session")


class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestContract:
    def test_turn_roundtrip(self, client):
        """Write a turn, recall it, verify shape."""
        r = client.post(
            "/turns",
            json={
                "session_id": "test-contract-session",
                "user_id": "test-contract-user",
                "messages": [
                    {"role": "user", "content": "I work at Google as a designer."},
                    {
                        "role": "assistant",
                        "content": "Nice! How long have you been there?",
                    },
                ],
                "timestamp": "2025-05-01T10:00:00Z",
                "metadata": {},
            },
        )
        assert r.status_code == 201
        body = r.json()
        assert "id" in body

        # Recall should find it
        r = client.post(
            "/recall",
            json={
                "query": "Where does the user work?",
                "session_id": "test-contract-session",
                "user_id": "test-contract-user",
                "max_tokens": 512,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "context" in body
        assert "citations" in body
        assert "Google" in body["context"] or "designer" in body["context"]

    def test_recall_empty_session(self, client):
        """Cold session returns empty context, not error."""
        r = client.post(
            "/recall",
            json={
                "query": "anything",
                "session_id": "nonexistent-session",
                "user_id": "nonexistent-user",
                "max_tokens": 512,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["context"] == ""
        assert body["citations"] == []

    def test_search_endpoint(self, client):
        """Search returns structured results."""
        client.post(
            "/turns",
            json={
                "session_id": "test-contract-session",
                "user_id": "test-contract-user",
                "messages": [
                    {"role": "user", "content": "My favorite color is blue."},
                    {"role": "assistant", "content": "Blue is a great color!"},
                ],
                "timestamp": "2025-05-01T11:00:00Z",
                "metadata": {},
            },
        )

        r = client.post(
            "/search",
            json={
                "query": "favorite color",
                "user_id": "test-contract-user",
                "limit": 5,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "results" in body
        assert isinstance(body["results"], list)
        assert body["results"]
        assert any("blue" in result["content"].lower() for result in body["results"])

    def test_user_memories_endpoint(self, client):
        """User memories returns structured data."""
        client.post(
            "/turns",
            json={
                "session_id": "test-contract-session",
                "user_id": "test-contract-user",
                "messages": [
                    {"role": "user", "content": "I have a cat named Whiskers."},
                    {"role": "assistant", "content": "Cute name!"},
                ],
                "timestamp": "2025-05-01T12:00:00Z",
                "metadata": {},
            },
        )

        r = client.get("/users/test-contract-user/memories")
        assert r.status_code == 200
        body = r.json()
        assert "memories" in body
        assert isinstance(body["memories"], list)
        if body["memories"]:
            m = body["memories"][0]
            assert "id" in m
            assert "type" in m
            assert "key" in m
            assert "value" in m
            assert "active" in m

    def test_delete_session(self, client):
        """Delete session returns 204."""
        r = client.delete("/sessions/test-contract-session")
        assert r.status_code == 204

    def test_delete_user(self, client):
        """Delete user returns 204."""
        r = client.delete("/users/test-contract-user")
        assert r.status_code == 204


class TestMalformedInput:
    def test_invalid_json(self, client):
        r = client.post("/turns", content=b"not json", headers={"content-type": "application/json"})
        assert r.status_code == 422

    def test_missing_fields(self, client):
        r = client.post("/turns", json={"session_id": "x"})
        assert r.status_code == 422

    def test_invalid_recall_budget(self, client):
        r = client.post(
            "/recall",
            json={
                "query": "anything",
                "session_id": "test-contract-session",
                "user_id": "test-contract-user",
                "max_tokens": 0,
            },
        )
        assert r.status_code == 422

    def test_invalid_search_limit(self, client):
        r = client.post(
            "/search",
            json={
                "query": "anything",
                "user_id": "test-contract-user",
                "limit": 0,
            },
        )
        assert r.status_code == 422

    def test_unicode_content(self, client):
        """Unicode doesn't crash the service."""
        r = client.post(
            "/turns",
            json={
                "session_id": "test-contract-session",
                "user_id": "test-contract-user",
                "messages": [
                    {
                        "role": "user",
                        "content": "I like 日本語 and émojis 🎉🚀 and Ñoño",
                    },
                    {"role": "assistant", "content": "Cool! 你好世界"},
                ],
                "timestamp": "2025-05-01T10:00:00Z",
                "metadata": {},
            },
        )
        assert r.status_code == 201

    def test_empty_messages(self, client):
        r = client.post(
            "/turns",
            json={
                "session_id": "test-contract-session",
                "user_id": "test-contract-user",
                "messages": [],
                "timestamp": "2025-05-01T10:00:00Z",
                "metadata": {},
            },
        )
        # Should handle gracefully (either 201 with no extraction or 422)
        assert r.status_code in (201, 422)


class TestConcurrentSessions:
    def test_no_bleed_between_users(self, client):
        """Two users' memories don't bleed."""
        client.post(
            "/turns",
            json={
                "session_id": "session-user-a",
                "user_id": "user-a",
                "messages": [
                    {"role": "user", "content": "I'm a doctor in London."},
                    {"role": "assistant", "content": "Interesting!"},
                ],
                "timestamp": "2025-05-01T10:00:00Z",
                "metadata": {},
            },
        )
        client.post(
            "/turns",
            json={
                "session_id": "session-user-b",
                "user_id": "user-b",
                "messages": [
                    {"role": "user", "content": "I'm a chef in Tokyo."},
                    {"role": "assistant", "content": "Cool!"},
                ],
                "timestamp": "2025-05-01T10:00:00Z",
                "metadata": {},
            },
        )

        # User A recall should not mention Tokyo/chef
        r = client.post(
            "/recall",
            json={
                "query": "What does this user do?",
                "session_id": "session-user-a",
                "user_id": "user-a",
                "max_tokens": 512,
            },
        )
        context = r.json()["context"]
        assert "Tokyo" not in context
        assert "chef" not in context

        # Cleanup
        client.delete("/users/user-a")
        client.delete("/users/user-b")


class TestRecallQuality:
    """Run the fixture-based recall quality test."""

    def test_fixture_recall(self, client):
        with open("fixtures/conversations.json") as f:
            fixture = json.load(f)

        # Ingest all conversations
        for conv in fixture["conversations"]:
            for turn in conv["turns"]:
                r = client.post("/turns", json=turn)
                assert r.status_code == 201

        # Run probes
        passed = 0
        total = len(fixture["probes"])

        for probe in fixture["probes"]:
            r = client.post(
                "/recall",
                json={
                    "query": probe["query"],
                    "session_id": probe["session_id"],
                    "user_id": probe["user_id"],
                    "max_tokens": 1024,
                },
            )
            assert r.status_code == 200
            context = r.json()["context"].lower()

            # Check expected facts
            found = sum(1 for fact in probe["expected_facts"] if fact.lower() in context)
            if found == len(probe["expected_facts"]):
                passed += 1
            else:
                missing = [f for f in probe["expected_facts"] if f.lower() not in context]
                print(f"  MISS: '{probe['query']}' missing: {missing}")

        score = passed / total
        print(f"\nRecall quality: {passed}/{total} probes passed ({score:.0%})")
        # We want at least 50% of probes to pass
        assert score >= 0.5, f"Recall quality too low: {score:.0%}"

        # Cleanup
        client.delete("/users/test-user-1")
