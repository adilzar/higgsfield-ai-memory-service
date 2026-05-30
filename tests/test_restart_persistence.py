import json
import os
import subprocess
import time
import unittest
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.getenv("MEMORY_BASE_URL", "http://localhost:8080")
REPO_ROOT = Path(__file__).resolve().parents[1]


def request(
    method: str,
    path: str,
    payload: dict | None = None,
    timeout: int = 20,
) -> tuple[int, dict | str]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"content-type": "application/json"} if payload is not None else {}
    req = Request(f"{BASE_URL}{path}", data=body, headers=headers, method=method)

    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            if not raw:
                return response.status, ""
            return response.status, json.loads(raw)
    except HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


def wait_for_health(timeout_seconds: int = 90) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            status, body = request("GET", "/health", timeout=2)
            if status == 200 and isinstance(body, dict) and body.get("status") == "ok":
                return
        except (OSError, TimeoutError, URLError) as exc:
            last_error = exc
        time.sleep(1)

    raise AssertionError(f"service did not become healthy after restart: {last_error}")


def cleanup_user(user_id: str) -> None:
    try:
        wait_for_health(timeout_seconds=30)
        request("DELETE", f"/users/{user_id}", timeout=10)
    except (OSError, TimeoutError, URLError, AssertionError):
        pass


@unittest.skipUnless(
    os.getenv("RUN_RESTART_TESTS") == "1",
    "set RUN_RESTART_TESTS=1 to run Docker restart persistence test",
)
class RestartPersistenceTest(unittest.TestCase):
    def test_memories_survive_service_restart(self) -> None:
        unique = uuid.uuid4().hex
        user_id = f"restart-user-{unique}"
        intake_session_id = f"restart-intake-{unique}"
        recall_session_id = f"restart-recall-{unique}"
        fact_token = f"Kelvinport-{unique[:8]}"

        self.addCleanup(cleanup_user, user_id)

        wait_for_health()
        self.ingest_fact(user_id, intake_session_id, fact_token)
        self.assert_memory_contains(user_id, fact_token)

        subprocess.run(
            ["docker", "compose", "restart", "service"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        wait_for_health()

        status, body = request("POST", "/recall", {
            "query": "Where does this user live?",
            "session_id": recall_session_id,
            "user_id": user_id,
            "max_tokens": 512,
        })

        self.assertEqual(status, 200, body)
        self.assertIsInstance(body, dict)
        self.assertIn(
            fact_token,
            body["context"],
            "post-restart recall missed the persisted cross-session memory",
        )

    def ingest_fact(self, user_id: str, session_id: str, fact_token: str) -> None:
        status, body = request("POST", "/turns", {
            "session_id": session_id,
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": f"I live in {fact_token}. Please remember that exact city name.",
                },
                {"role": "assistant", "content": "Got it."},
            ],
            "timestamp": "2026-05-30T10:00:00Z",
            "metadata": {"test": "restart-persistence"},
        })

        self.assertEqual(status, 201, body)
        self.assertIsInstance(body, dict)
        self.assertIn("id", body)

    def assert_memory_contains(self, user_id: str, fact_token: str) -> None:
        status, body = request("GET", f"/users/{user_id}/memories")

        self.assertEqual(status, 200, body)
        self.assertIsInstance(body, dict)
        values = [m["value"] for m in body["memories"]]
        self.assertTrue(
            any(fact_token in value for value in values),
            f"pre-restart extraction did not persist {fact_token!r}: {values}",
        )


if __name__ == "__main__":
    unittest.main()
