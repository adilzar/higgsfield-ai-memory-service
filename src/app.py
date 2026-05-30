from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.auth import enforce_memory_auth
from src.bootstrap import initialize_service
from src.routes import router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_service()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Memory Service", lifespan=lifespan)
    app.middleware("http")(enforce_memory_auth)
    app.include_router(router)
    return app
