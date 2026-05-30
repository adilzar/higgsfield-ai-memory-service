from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from src.api_models import (
    RecallRequest,
    SearchRequest,
    TurnRequest,
    UserMemoriesResponse,
    format_user_memories,
)
from src.auth import enforce_memory_auth
from src.bootstrap import initialize_service
from src.database import get_db
from src.intake import ingest_turn
from src.lifecycle import delete_session_data, delete_user_data
from src.recall import build_recall_context
from src.search import search_memories
from src.store import fetch_user_memory_models

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_service()
    yield


app = FastAPI(title="Memory Service", lifespan=lifespan)
app.middleware("http")(enforce_memory_auth)


# --- Endpoints ---
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/turns", status_code=201)
async def create_turn(req: TurnRequest, db: AsyncSession = Depends(get_db)):
    turn_id = await ingest_turn(db, req.to_command())
    return {"id": turn_id}


@app.post("/recall")
async def recall(req: RecallRequest, db: AsyncSession = Depends(get_db)):
    context, citations = await build_recall_context(db, req.to_command())
    return {"context": context, "citations": citations}


@app.post("/search")
async def search(req: SearchRequest, db: AsyncSession = Depends(get_db)):
    results = await search_memories(db, req.to_command())
    return {"results": results}


@app.get("/users/{user_id}/memories", response_model=UserMemoriesResponse)
async def get_user_memories(user_id: str, db: AsyncSession = Depends(get_db)):
    memories = await fetch_user_memory_models(db, user_id)
    return format_user_memories(memories)


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    await delete_session_data(db, session_id)
    return None


@app.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: str, db: AsyncSession = Depends(get_db)):
    await delete_user_data(db, user_id)
    return None
