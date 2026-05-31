from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    RecallRequest,
    SearchRequest,
    TurnRequest,
    UserMemoriesResponse,
    format_user_memories,
)
from src.core.lifecycle import delete_session_data, delete_user_data
from src.core.search import search_memories
from src.ingestion.intake import ingest_turn
from src.recall import RecallContext, build_recall_context
from src.storage.database import get_db
from src.storage.store import fetch_user_memory_models

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/turns", status_code=201)
async def create_turn(req: TurnRequest, db: AsyncSession = Depends(get_db)):
    turn_id = await ingest_turn(db, req.to_command())
    return {"id": turn_id}


@router.post("/recall")
async def recall(req: RecallRequest, db: AsyncSession = Depends(get_db)):
    context = await build_recall_context(db, req.to_command())
    return format_recall_response(context)


@router.post("/search")
async def search(req: SearchRequest, db: AsyncSession = Depends(get_db)):
    results = await search_memories(db, req.to_command())
    return {"results": results}


@router.get("/users/{user_id}/memories", response_model=UserMemoriesResponse)
async def get_user_memories(user_id: str, db: AsyncSession = Depends(get_db)):
    memories = await fetch_user_memory_models(db, user_id)
    return format_user_memories(memories)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    await delete_session_data(db, session_id)
    return None


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: str, db: AsyncSession = Depends(get_db)):
    await delete_user_data(db, user_id)
    return None


def format_recall_response(context: RecallContext) -> dict:
    return {
        "context": context.text,
        "citations": [
            {
                "turn_id": citation.turn_id,
                "score": citation.score,
                "snippet": citation.snippet,
            }
            for citation in context.citations
        ],
    }
