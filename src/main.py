from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database import get_db, init_db
from src.embeddings import embed_text
from src.intake import IngestTurnCommand, TurnMessage, ingest_turn
from src.recall import build_recall_context
from src.search import SearchMemoriesCommand, search_memories
from src.store import delete_session_data, delete_user_data, fetch_user_memory_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Warm up embedding model
    embed_text("warmup")
    logger.info("Memory service ready")
    yield


app = FastAPI(title="Memory Service", lifespan=lifespan)


# --- Auth middleware ---
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if settings.memory_auth_token:
        auth = request.headers.get("authorization", "")
        if request.url.path != "/health" and auth != f"Bearer {settings.memory_auth_token}":
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


# --- Request/Response Models ---
class Message(BaseModel):
    role: str
    content: str
    name: str | None = None


class TurnRequest(BaseModel):
    session_id: str
    user_id: str | None = None
    messages: list[Message]
    timestamp: str | None = None
    metadata: dict | None = None


class RecallRequest(BaseModel):
    query: str
    session_id: str
    user_id: str | None = None
    max_tokens: int = Field(default=1024, gt=0)


class SearchRequest(BaseModel):
    query: str
    session_id: str | None = None
    user_id: str | None = None
    limit: int = Field(default=10, gt=0)


# --- Endpoints ---
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/turns", status_code=201)
async def create_turn(req: TurnRequest, db: AsyncSession = Depends(get_db)):
    turn_id = await ingest_turn(
        db,
        IngestTurnCommand(
            session_id=req.session_id,
            user_id=req.user_id,
            messages=[TurnMessage(m.role, m.content, m.name) for m in req.messages],
            timestamp=req.timestamp,
            metadata=req.metadata,
        ),
    )
    return {"id": turn_id}


@app.post("/recall")
async def recall(req: RecallRequest, db: AsyncSession = Depends(get_db)):
    context, citations = await build_recall_context(
        db, req.query, req.user_id, req.session_id, req.max_tokens
    )
    return {"context": context, "citations": citations}


@app.post("/search")
async def search(req: SearchRequest, db: AsyncSession = Depends(get_db)):
    results = await search_memories(
        db,
        SearchMemoriesCommand(
            query=req.query,
            user_id=req.user_id,
            session_id=req.session_id,
            limit=req.limit,
        ),
    )
    return {"results": results}


@app.get("/users/{user_id}/memories")
async def get_user_memories(user_id: str, db: AsyncSession = Depends(get_db)):
    memories = await fetch_user_memory_models(db, user_id)
    return {
        "memories": [
            {
                "id": m.id,
                "type": m.type,
                "key": m.key,
                "value": m.value,
                "confidence": m.confidence,
                "source_session": m.session_id,
                "source_turn": m.source_turn_id,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                "supersedes": m.supersedes,
                "superseded_by": m.superseded_by,
                "active": m.active,
            }
            for m in memories
        ]
    }


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    await delete_session_data(db, session_id)
    return None


@app.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: str, db: AsyncSession = Depends(get_db)):
    await delete_user_data(db, user_id)
    return None
