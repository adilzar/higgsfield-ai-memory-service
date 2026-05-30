from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import delete, select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database import get_db, init_db
from src.embeddings import embed_text
from src.intake import IngestTurnCommand, TurnMessage, ingest_turn
from src.models import Memory, Turn
from src.recall import build_recall_context

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
    max_tokens: int = 1024


class SearchRequest(BaseModel):
    query: str
    session_id: str | None = None
    user_id: str | None = None
    limit: int = 10


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
    query_embedding = embed_text(req.query)

    conditions = ["active = true"]
    params: dict = {"embedding": str(query_embedding), "limit": req.limit, "query": req.query}

    if req.user_id:
        conditions.append("user_id = :user_id")
        params["user_id"] = req.user_id
    if req.session_id:
        conditions.append("session_id = :session_id")
        params["session_id"] = req.session_id

    where = " AND ".join(conditions)

    sql = sa_text(f"""
        SELECT id, value, confidence, session_id, created_at, type, key,
               1 - (embedding <=> CAST(:embedding AS vector)) as score
        FROM memories
        WHERE {where}
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)
    result = await db.execute(sql, params)
    rows = result.mappings().all()

    return {
        "results": [
            {
                "content": r["value"],
                "score": round(float(r["score"]), 4),
                "session_id": r["session_id"],
                "timestamp": r["created_at"].isoformat() if r["created_at"] else None,
                "metadata": {"type": r["type"], "key": r["key"]},
            }
            for r in rows
        ]
    }


@app.get("/users/{user_id}/memories")
async def get_user_memories(user_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Memory).where(Memory.user_id == user_id).order_by(Memory.created_at.desc())
    )
    memories = result.scalars().all()
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
    await db.execute(delete(Memory).where(Memory.session_id == session_id))
    await db.execute(delete(Turn).where(Turn.session_id == session_id))
    await db.commit()
    return None


@app.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: str, db: AsyncSession = Depends(get_db)):
    # Get all sessions for this user to clean up turns
    result = await db.execute(select(Turn.session_id).where(Turn.user_id == user_id).distinct())
    session_ids = [r[0] for r in result.all()]

    await db.execute(delete(Memory).where(Memory.user_id == user_id))
    await db.execute(delete(Turn).where(Turn.user_id == user_id))
    await db.commit()
    return None
