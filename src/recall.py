import logging
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from src.embeddings import embed_text

logger = logging.getLogger(__name__)

RRF_K = 60


async def hybrid_retrieve(
    db: AsyncSession, query: str, user_id: str | None, session_id: str, limit: int = 20
) -> list[dict]:
    """Hybrid retrieval: vector similarity + full-text search, fused with RRF."""
    query_embedding = embed_text(query)

    # Vector search on memories
    vector_sql = sa_text("""
        SELECT id, type, key, value, confidence, session_id, source_turn_id, created_at, updated_at,
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

    # Sort by fused score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for mid, score in ranked:
        entry = memory_map[mid]
        entry["rrf_score"] = score
        results.append(entry)

    return results


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
        if m.get("updated_at"):
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
