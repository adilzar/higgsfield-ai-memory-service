from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import Request
from fastapi.responses import JSONResponse

from src.core.config import settings

PUBLIC_PATHS = frozenset({"/health"})


@dataclass(frozen=True)
class AuthPolicy:
    token: str = ""
    public_paths: frozenset[str] = field(default=PUBLIC_PATHS)

    def is_authorized(self, path: str, authorization: str | None) -> bool:
        if path in self.public_paths:
            return True
        if not self.token:
            return True
        return authorization == f"Bearer {self.token}"


async def enforce_memory_auth(request: Request, call_next):
    policy = AuthPolicy(settings.memory_auth_token)
    if not policy.is_authorized(
        request.url.path,
        request.headers.get("authorization"),
    ):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)
