from src.api.app import create_app
from src.api.auth import enforce_memory_auth


def test_create_app_registers_expected_routes():
    app = create_app()

    routes = {(route.path, ",".join(sorted(route.methods))) for route in app.routes}

    assert ("/health", "GET") in routes
    assert ("/turns", "POST") in routes
    assert ("/recall", "POST") in routes
    assert ("/search", "POST") in routes
    assert ("/users/{user_id}/memories", "GET") in routes
    assert ("/sessions/{session_id}", "DELETE") in routes
    assert ("/users/{user_id}", "DELETE") in routes


def test_create_app_registers_auth_middleware():
    app = create_app()

    dispatches = [middleware.kwargs.get("dispatch") for middleware in app.user_middleware]

    assert enforce_memory_auth in dispatches
