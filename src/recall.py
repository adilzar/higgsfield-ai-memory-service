from __future__ import annotations

import logging
import re

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from src.embeddings import embed_text

logger = logging.getLogger(__name__)

RRF_K = 60

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
    "employment": {"work", "works", "job", "career", "employer", "company", "role", "pm", "engineer", "designer"},
    "pet": {"dog", "cat", "pet", "pets", "biscuit"},
    "diet": {"diet", "dietary", "vegetarian", "shellfish", "allergy", "allergic", "food", "eat"},
    "communication_style": {"concise", "direct", "answers", "style", "prefer", "preference", "fluff"},
    "opinion": {"think", "opinion", "programming", "language", "typescript", "python", "love"},
}

STABLE_INTENTS = {"location", "employment", "pet", "diet", "communication_style"}
HISTORY_TERMS = {"history", "previous", "previously", "past", "former", "before"}


async def hybrid_retrieve(
    db: AsyncSession, query: str, user_id: str | None, session_id: str, limit: int = 20
) -> list[dict]:
    """Hybrid retrieval: vector similarity + full-text search, fused with RRF."""
    query_embedding = embed_text(query)

    # Vector search on memories
    vector_sql = sa_text("""
        SELECT id, type, key, value, confidence, session_id, source_turn_id, created_at, updated_at,
               active, supersedes, superseded_by,
               1 - (embedding <=> CAST(:embedding AS vector)) as vec_score
        FROM memories
        WHERE active = true
          AND (user_id = :user_id OR (:user_id IS NULL AND session_id = :session_id))
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)
    vec_result = await db.execute(vector_sql, {
        "embedding": str(query_embedding), "user_id": user_id,
        "session_id": session_id, "limit": limit
    })
    vec_rows = vec_result.mappings().all()

    # Full-text search on memories
    fts_sql = sa_text("""
        SELECT id, type, key, value, confidence, session_id, source_turn_id, created_at, updated_at,
               active, supersedes, superseded_by,
               ts_rank(to_tsvector('english', value), plainto_tsquery('english', :query)) as fts_score
        FROM memories
        WHERE active = true
          AND (user_id = :user_id OR (:user_id IS NULL AND session_id = :session_id))
          AND to_tsvector('english', value) @@ plainto_tsquery('english', :query)
        ORDER BY fts_score DESC
        LIMIT :limit
    """)
    fts_result = await db.execute(fts_sql, {
        "query": query, "user_id": user_id, "session_id": session_id, "limit": limit
    })
    fts_rows = fts_result.mappings().all()

    # RRF fusion
    scores: dict[str, float] = {}
    memory_map: dict[str, dict] = {}

    for rank, row in enumerate(vec_rows):
        mid = row["id"]
        scores[mid] = scores.get(mid, 0) + 1.0 / (RRF_K + rank + 1)
        memory_map[mid] = dict(row)

    for rank, row in enumerate(fts_rows):
        mid = row["id"]
        scores[mid] = scores.get(mid, 0) + 1.0 / (RRF_K + rank + 1)
        if mid not in memory_map:
            memory_map[mid] = dict(row)
        else:
            memory_map[mid]["fts_score"] = row["fts_score"]

    # Sort by fused score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for mid, score in ranked:
        entry = memory_map[mid]
        entry["rrf_score"] = score
        results.append(entry)

    return results


async def get_scope_memories(
    db: AsyncSession, user_id: str | None, session_id: str, include_inactive: bool = False
) -> list[dict]:
    """Fetch memories available for Recall expansion."""
    active_clause = "" if include_inactive else "AND active = true"
    sql = sa_text(f"""
        SELECT id, type, key, value, confidence, session_id, source_turn_id, created_at, updated_at,
               active, supersedes, superseded_by
        FROM memories
        WHERE (user_id = :user_id OR (:user_id IS NULL AND session_id = :session_id))
          {active_clause}
        ORDER BY active DESC, created_at DESC
    """)
    result = await db.execute(sql, {"user_id": user_id, "session_id": session_id})
    return [dict(r) for r in result.mappings().all()]


async def build_recall_context(
    db: AsyncSession,
    query: str,
    user_id: str | None,
    session_id: str,
    max_tokens: int,
) -> tuple[str, list[dict]]:
    """Build prompt Context and Citations for /recall."""
    include_inactive = _needs_history(query)
    retrieved = await hybrid_retrieve(db, query, user_id, session_id)
    scope_memories = await get_scope_memories(db, user_id, session_id, include_inactive)
    memories = _select_recall_memories(query, retrieved, scope_memories)

    if not memories:
        return "", []

    recent = await get_recent_turns(db, session_id)
    return assemble_context(memories, recent, max_tokens)


async def get_recent_turns(
    db: AsyncSession, session_id: str, limit: int = 5
) -> list[dict]:
    """Get recent turns from the current session for conversational context."""
    sql = sa_text("""
        SELECT id, content_text, timestamp
        FROM turns
        WHERE session_id = :session_id
        ORDER BY timestamp DESC
        LIMIT :limit
    """)
    result = await db.execute(sql, {"session_id": session_id, "limit": limit})
    return [dict(r) for r in result.mappings().all()]


def assemble_context(
    memories: list[dict], recent_turns: list[dict], max_tokens: int
) -> tuple[str, list[dict]]:
    """Assemble context string with tiered priority, respecting token budget."""
    # Approximate tokens as words * 1.3
    def estimate_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)

    budget = max_tokens
    sections = []
    citations = []

    # Tier 1: Stable facts
    facts = [m for m in memories if m["type"] == "fact"]
    # Tier 2: Preferences
    preferences = [m for m in memories if m["type"] == "preference"]
    # Tier 3: Other query-relevant (opinions, events)
    others = [m for m in memories if m["type"] in ("opinion", "event")]

    # Build facts section
    fact_lines = []
    for m in facts[:15]:  # cap at 15 facts
        line = f"- {m['value']}"
        if m.get("active") is False:
            line = f"- Previously: {m['value']}"
        elif m.get("updated_at"):
            line += f" (updated {str(m['updated_at'])[:10]})"
        cost = estimate_tokens(line)
        if budget - cost < 0:
            break
        budget -= cost
        fact_lines.append(line)
        citations.append({"turn_id": m["source_turn_id"], "score": m.get("rrf_score", 0), "snippet": m["value"]})

    if fact_lines:
        sections.append("## Known facts about this user\n" + "\n".join(fact_lines))

    # Build preferences section
    pref_lines = []
    for m in preferences[:10]:
        line = f"- {m['value']}"
        cost = estimate_tokens(line)
        if budget - cost < 0:
            break
        budget -= cost
        pref_lines.append(line)
        citations.append({"turn_id": m["source_turn_id"], "score": m.get("rrf_score", 0), "snippet": m["value"]})

    if pref_lines:
        sections.append("## Preferences\n" + "\n".join(pref_lines))

    # Build relevant memories section
    other_lines = []
    for m in others[:10]:
        line = f"- [{str(m.get('created_at', ''))[:10]}] {m['value']}"
        cost = estimate_tokens(line)
        if budget - cost < 0:
            break
        budget -= cost
        other_lines.append(line)
        citations.append({"turn_id": m["source_turn_id"], "score": m.get("rrf_score", 0), "snippet": m["value"]})

    if other_lines:
        sections.append("## Relevant memories\n" + "\n".join(other_lines))

    # Tier 4: Recent conversation context
    recent_lines = []
    for t in recent_turns:
        text = t["content_text"][:200]
        line = f"- [{str(t.get('timestamp', ''))[:10]}] {text}"
        cost = estimate_tokens(line)
        if budget - cost < 0:
            break
        budget -= cost
        recent_lines.append(line)
        citations.append({"turn_id": t["id"], "score": 0.0, "snippet": text[:100]})

    if recent_lines:
        sections.append("## Recent conversation context\n" + "\n".join(recent_lines))

    context = "\n\n".join(sections)
    return context, citations


def _select_recall_memories(query: str, retrieved: list[dict], scope_memories: list[dict]) -> list[dict]:
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
        memory for memory in retrieved
        if _memory_matches_query(memory, query_tokens, query_intents)
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


def _memory_matches_query(memory: dict, query_tokens: set[str], query_intents: set[str]) -> bool:
    if memory.get("fts_score", 0) and float(memory["fts_score"]) > 0:
        return True

    memory_tokens = _tokens(" ".join([
        str(memory.get("key", "")),
        str(memory.get("type", "")),
        str(memory.get("value", "")),
    ]))
    if query_tokens & memory_tokens:
        return True

    return bool(query_intents & _memory_intents(memory))


def _query_intents(query: str) -> set[str]:
    tokens = _tokens(query)
    intents = {
        intent
        for intent, terms in INTENT_TERMS.items()
        if tokens & terms
    }
    if "preference" in intents:
        intents.add("communication_style")
    return intents


def _memory_intents(memory: dict) -> set[str]:
    key = str(memory.get("key", "")).lower()
    memory_type = str(memory.get("type", "")).lower()
    intents: set[str] = set()

    if any(part in key for part in ("location", "city", "home", "moved")):
        intents.add("location")
    if any(part in key for part in ("employment", "career", "job", "work", "company")):
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
    return False


def _needs_history(query: str) -> bool:
    return bool(_tokens(query) & HISTORY_TERMS)


def _tokens(text: str) -> set[str]:
    return {token for token in _raw_tokens(text) if token not in STOPWORDS and len(token) > 1}


def _raw_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))
