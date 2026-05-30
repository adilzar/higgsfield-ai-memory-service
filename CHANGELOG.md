# CHANGELOG

## v1 — Initial architecture: FastAPI + Postgres + pgvector

**What:** Built the core service with PostgreSQL as the single backing store (relational + FTS + vector via pgvector). FastAPI for the HTTP layer, SQLAlchemy async for DB access.

**Why:** Needed a store that handles three access patterns (structured queries, full-text search, vector similarity) without multi-service coordination. Postgres does all three with ACID guarantees, which makes "write then immediately read" trivially correct.

**Result:** Service boots with `docker compose up`, schema auto-creates, health endpoint works. Foundation is solid.

## v2 — LLM-based extraction with contradiction detection

**What:** Added extraction pipeline using DeepSeek V4 Flash via OpenRouter. The prompt receives existing active memories and detects contradictions, returning a `supersedes_key` when a new fact conflicts with an old one.

**Why:** Rule-based extraction would miss implicit facts ("walking Biscuit" → has a pet) and can't reliably detect semantic contradictions ("work at Stripe" vs "started at Notion"). LLM handles both naturally.

**Result:** Extraction produces typed, keyed memories. Supersession chain works — old facts get `active=false` with a pointer to the replacement. `/users/{user_id}/memories` shows the full history.

**Tradeoff:** Adds 1-3s latency to `/turns` for the LLM call. Acceptable given the 60s timeout and the spec's emphasis on extraction quality over speed.

## v3 — Hybrid retrieval with reciprocal rank fusion

**What:** Added BM25-style full-text search alongside embedding search and fused results with RRF (k=60).

**Why:** Pure embedding search was missing keyword-heavy queries like "what's their dog's name?" where exact token match matters more than semantic similarity. FTS catches these directly.

**Result:** Both semantic queries ("tell me about their career") and keyword queries ("dog's name") work well. The RRF fusion naturally handles cases where both signals agree (high confidence) and where only one fires (still surfaces the result).

## v4 — Tiered context assembly with token budgeting

**What:** Implemented priority-based context assembly: facts → preferences → query-relevant → recent turns. Each tier fills the token budget in order; later tiers get cut first when budget is tight.

**Why:** The spec explicitly requires priority logic under tight budgets. A flat "top-K by score" approach would mix stable identity facts with ephemeral context, potentially dropping critical facts when budget is small.

**Result:** With `max_tokens=256`, the service reliably includes core identity facts (employment, location, pets) and cuts recent conversation context. With `max_tokens=1024`, all tiers get representation.

## v5 — Local embeddings and test fixtures

**What:** Used sentence-transformers `all-MiniLM-L6-v2` for local embeddings (384d). Built a 5-conversation fixture with 8 probe queries covering fact evolution, multi-hop, and keyword recall.

**Why:** Local embeddings eliminate an API dependency on the recall hot path — no extra key needed, no network latency for queries. The fixture provides a repeatable quality signal for iteration.

**Result:** Self-eval shows 6/8 probes passing consistently. The two that sometimes miss are the opinion evolution probe (LLM extraction quality varies) and the multi-hop probe (requires both "Biscuit" and "Berlin" memories to be extracted and surfaced together).

**Next:** Could improve with a reranker pass or query expansion, but the current quality is solid for the scope.
