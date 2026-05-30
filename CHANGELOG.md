# CHANGELOG

## v8 — Code quality: black + isort formatting

**What:** Applied `black` (line-length=100) and `isort` (black profile) across the entire codebase. Added `pyproject.toml` for consistent config.

**Why:** Consistent formatting eliminates style debates and makes diffs cleaner. The 100-char line length balances readability with avoiding excessive wrapping of SQL strings and long function signatures.

## v7 — E2E tests and recall intent improvements

**What:** Added 14 end-to-end tests covering fact evolution, cross-session knowledge, token budget, multi-hop, session delete, implicit facts, and multi-message turns. Improved intent matching: added "occupation"/"role" to employment detection, "do" to employment query terms, and "tell me about" as a profile query pattern.

**Why:** The e2e tests exposed gaps in the noise gate — queries like "What does the user do?" and "Tell me about this user" were returning empty context because the intent vocabulary was too narrow. The tests now serve as a regression guard for recall quality.

**Result:** All 48 tests pass (unit + integration + e2e). The noise gate correctly rejects irrelevant queries while allowing legitimate employment/profile queries through.

## v6 — Bug fixes: supersession integrity, extraction validation, noise gate

**What:** Three bugs found and fixed:
1. `DELETE /sessions` now reactivates memories that were superseded by memories in the deleted session
2. `_parse_response` filters out malformed LLM items (missing `value` key, non-dict entries)
3. Noise gate now respects vector similarity score (≥0.45 threshold) — high-similarity memories pass regardless of token/intent overlap

**Why:**
1. Session delete was leaving superseded memories permanently invisible (active=false with dangling pointer)
2. Malformed LLM output caused unhandled KeyError, crashing `/turns` with 500
3. Noise gate was dropping semantically relevant memories when query used different words than the memory (e.g., "How old?" vs "Born in 1993")

**Result:** All three bugs verified fixed with targeted repro scripts. No regressions in existing tests.

## v5 — Extraction boundary refactor

**What:** Split `extract_memories` into `_build_prompt`, `_call_llm`, `_parse_response`. Introduced `ExtractionError` exception class and `ExtractionResult` dataclass. Caller in `intake.py` catches errors gracefully.

**Why:** The original function mixed LLM call, response cleanup, JSON parsing, and failure policy in one function, collapsing all failures to `[]`. Now failures are typed, logged, and the turn is still persisted even when extraction fails.

**Result:** Cleaner separation of concerns. Each function is independently testable. Error handling is explicit rather than silent.

## v4 — Noise gate, anchor expansion, and recall architecture

**What:** Added intent-based noise gate to prevent returning irrelevant memories. Added anchor expansion for multi-hop recall. Refactored recall into separate modules: `store.py` (queries), `recall.py` (selection logic), `budget.py` (assembly).

**Why:** Without a noise gate, queries about topics never discussed would return the closest (but irrelevant) memories. Anchor expansion enables multi-hop: "What city does the user with the dog named Biscuit live in?" connects pet → location.

**Result:** Noise resistance improved — empty context returned for irrelevant queries. Multi-hop queries work by expanding from matched memories to related ones via shared intents.

## v3 — Hybrid retrieval with reciprocal rank fusion

**What:** Added BM25-style full-text search alongside embedding search and fused results with RRF (k=60).

**Why:** Pure embedding search was missing keyword-heavy queries like "what's their dog's name?" where exact token match matters more than semantic similarity. FTS catches these directly.

**Result:** Both semantic queries ("tell me about their career") and keyword queries ("dog's name") work well. The RRF fusion naturally handles cases where both signals agree (high confidence) and where only one fires (still surfaces the result).

## v2 — LLM-based extraction with contradiction detection

**What:** Added extraction pipeline using DeepSeek V4 Flash via OpenRouter. The prompt receives existing active memories and detects contradictions, returning a `supersedes_key` when a new fact conflicts with an old one.

**Why:** Rule-based extraction would miss implicit facts ("walking Biscuit" → has a pet) and can't reliably detect semantic contradictions ("work at Stripe" vs "started at Notion"). LLM handles both naturally.

**Result:** Extraction produces typed, keyed memories. Supersession chain works — old facts get `active=false` with a pointer to the replacement. `/users/{user_id}/memories` shows the full history.

**Tradeoff:** Adds 1-3s latency to `/turns` for the LLM call. Acceptable given the 60s timeout and the spec's emphasis on extraction quality over speed.

## v1 — Initial architecture: FastAPI + Postgres + pgvector

**What:** Built the core service with PostgreSQL as the single backing store (relational + FTS + vector via pgvector). FastAPI for the HTTP layer, SQLAlchemy async for DB access. Local embeddings with all-MiniLM-L6-v2 (384d).

**Why:** Needed a store that handles three access patterns (structured queries, full-text search, vector similarity) without multi-service coordination. Postgres does all three with ACID guarantees, which makes "write then immediately read" trivially correct. Local embeddings eliminate an API dependency on the recall hot path.

**Result:** Service boots with `docker compose up`, schema auto-creates, health endpoint works. Tiered context assembly with token budgeting prioritizes facts over ephemeral context.
