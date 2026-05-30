from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from src.embeddings import embed_text
from src.recall.budget import assemble_context
from src.recall.retrieval import hybrid_search_memories
from src.store import fetch_recent_turns, fetch_scope_memories

logger = logging.getLogger(__name__)

STOPWORDS = {
    "about",
    "are",
    "does",
    "for",
    "from",
    "has",
    "have",
    "his",
    "her",
    "how",
    "into",
    "is",
    "its",
    "me",
    "my",
    "named",
    "of",
    "the",
    "their",
    "them",
    "this",
    "user",
    "was",
    "were",
    "what",
    "when",
    "who",
    "with",
}

INTENT_TERMS = {
    "location": {"where", "live", "lives", "city", "moved", "home", "reside", "living"},
    "employment": {
        "work",
        "works",
        "job",
        "career",
        "employer",
        "company",
        "role",
        "pm",
        "engineer",
        "designer",
        "do",
        "occupation",
    },
    "pet": {"dog", "cat", "pet", "pets", "biscuit"},
    "diet": {
        "diet",
        "dietary",
        "vegetarian",
        "shellfish",
        "allergy",
        "allergic",
        "food",
        "eat",
    },
    "communication_style": {
        "concise",
        "direct",
        "answers",
        "style",
        "prefer",
        "preference",
        "fluff",
    },
    "opinion": {
        "think",
        "opinion",
        "programming",
        "language",
        "typescript",
        "python",
        "love",
    },
}

STABLE_INTENTS = {"location", "employment", "pet", "diet", "communication_style"}
HISTORY_TERMS = {"history", "previous", "previously", "past", "former", "before"}


@dataclass(frozen=True)
class RecallContextCommand:
    query: str
    session_id: str
    user_id: str | None
    max_tokens: int


async def hybrid_retrieve(
    db: AsyncSession, query: str, user_id: str | None, session_id: str, limit: int = 20
) -> list[dict]:
    """Hybrid retrieval: vector similarity + full-text search, fused with RRF."""
    query_embedding = embed_text(query)
    return await hybrid_search_memories(db, query, query_embedding, user_id, session_id, limit)


async def build_recall_context(
    db: AsyncSession,
    cmd: RecallContextCommand,
) -> tuple[str, list[dict]]:
    """Build prompt Context and Citations for /recall."""
    include_inactive = _needs_history(cmd.query)
    retrieved = await hybrid_retrieve(db, cmd.query, cmd.user_id, cmd.session_id)
    scope_memories = await fetch_scope_memories(db, cmd.user_id, cmd.session_id, include_inactive)
    memories = _select_recall_memories(cmd.query, retrieved, scope_memories)

    if not memories:
        return "", []

    recent = await fetch_recent_turns(db, cmd.session_id)
    return assemble_context(memories, recent, cmd.max_tokens)


def _select_recall_memories(
    query: str, retrieved: list[dict], scope_memories: list[dict]
) -> list[dict]:
    query_tokens = _tokens(query)
    query_intents = _query_intents(query)
    profile_query = _is_profile_query(query, query_intents)
    include_history = _needs_history(query)
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add(memory: dict) -> None:
        mid = memory["id"]
        if mid not in selected_ids:
            selected.append(memory)
            selected_ids.add(mid)

    if profile_query:
        for memory in scope_memories:
            if memory.get("active") is True and _memory_intents(memory) & STABLE_INTENTS:
                add(memory)
        return selected

    relevant = [
        memory for memory in retrieved if _memory_matches_query(memory, query_tokens, query_intents)
    ]

    for memory in relevant:
        add(memory)

    if query_intents:
        for memory in scope_memories:
            if memory.get("active") is False:
                continue
            if _memory_intents(memory) & query_intents:
                add(memory)

    if not selected:
        return []

    expansion_intents = set(query_intents)
    for memory in selected:
        expansion_intents.update(_memory_intents(memory))

    if include_history:
        expansion_intents.update({"employment", "location", "opinion"})

    for memory in scope_memories:
        if memory.get("active") is False and not include_history:
            continue
        memory_intents = _memory_intents(memory)
        if memory_intents & expansion_intents:
            add(memory)

    return selected


VEC_SCORE_THRESHOLD = 0.45


def _memory_matches_query(memory: dict, query_tokens: set[str], query_intents: set[str]) -> bool:
    if memory.get("vec_score") and float(memory["vec_score"]) >= VEC_SCORE_THRESHOLD:
        return True

    if memory.get("fts_score", 0) and float(memory["fts_score"]) > 0:
        return True

    memory_tokens = _tokens(
        " ".join(
            [
                str(memory.get("key", "")),
                str(memory.get("type", "")),
                str(memory.get("value", "")),
            ]
        )
    )
    if query_tokens & memory_tokens:
        return True

    return bool(query_intents & _memory_intents(memory))


def _query_intents(query: str) -> set[str]:
    tokens = _tokens(query)
    intents = {intent for intent, terms in INTENT_TERMS.items() if tokens & terms}
    if "preference" in intents:
        intents.add("communication_style")
    return intents


def _memory_intents(memory: dict) -> set[str]:
    key = str(memory.get("key", "")).lower()
    memory_type = str(memory.get("type", "")).lower()
    intents: set[str] = set()

    if any(part in key for part in ("location", "city", "home", "moved")):
        intents.add("location")
    if any(
        part in key
        for part in (
            "employment",
            "career",
            "job",
            "work",
            "company",
            "occupation",
            "role",
        )
    ):
        intents.add("employment")
    if any(part in key for part in ("pet", "dog", "cat")):
        intents.add("pet")
    if any(part in key for part in ("food", "diet", "allerg", "shellfish")):
        intents.add("diet")
    if any(part in key for part in ("communication", "style", "answer")):
        intents.add("communication_style")
    if memory_type == "opinion":
        intents.add("opinion")
    if memory_type == "preference":
        intents.add("communication_style")

    return intents


def _is_profile_query(query: str, query_intents: set[str]) -> bool:
    tokens = _raw_tokens(query)
    if "profile" in tokens:
        return True
    if {"remember", "know"} & tokens and {"me", "user"} & tokens and not query_intents:
        return True
    if "everything" in tokens and {"me", "user"} & tokens:
        return True
    if {"tell", "about"} <= tokens and {"me", "user"} & tokens and not query_intents:
        return True
    return False


def _needs_history(query: str) -> bool:
    return bool(_tokens(query) & HISTORY_TERMS)


def _tokens(text: str) -> set[str]:
    return {token for token in _raw_tokens(text) if token not in STOPWORDS and len(token) > 1}


def _raw_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))
