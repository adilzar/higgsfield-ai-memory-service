from types import SimpleNamespace

import pytest

from src.api.auth import AuthPolicy, enforce_memory_auth


def test_auth_policy_allows_all_paths_when_token_disabled():
    policy = AuthPolicy(token="")

    assert policy.is_authorized("/recall", None) is True


def test_auth_policy_allows_public_paths_when_token_enabled():
    policy = AuthPolicy(token="secret")

    assert policy.is_authorized("/health", None) is True


def test_auth_policy_requires_matching_bearer_token():
    policy = AuthPolicy(token="secret")

    assert policy.is_authorized("/recall", None) is False
    assert policy.is_authorized("/recall", "Bearer wrong") is False
    assert policy.is_authorized("/recall", "Bearer secret") is True


@pytest.mark.asyncio
async def test_auth_middleware_rejects_unauthorized_request(monkeypatch):
    monkeypatch.setattr("src.api.auth.settings.memory_auth_token", "secret")
    request = SimpleNamespace(url=SimpleNamespace(path="/recall"), headers={})
    called = False

    async def call_next(_request):
        nonlocal called
        called = True
        return SimpleNamespace(status_code=200)

    response = await enforce_memory_auth(request, call_next)

    assert response.status_code == 401
    assert called is False


@pytest.mark.asyncio
async def test_auth_middleware_forwards_authorized_request(monkeypatch):
    monkeypatch.setattr("src.api.auth.settings.memory_auth_token", "secret")
    request = SimpleNamespace(
        url=SimpleNamespace(path="/recall"),
        headers={"authorization": "Bearer secret"},
    )

    async def call_next(_request):
        return SimpleNamespace(status_code=200)

    response = await enforce_memory_auth(request, call_next)

    assert response.status_code == 200
