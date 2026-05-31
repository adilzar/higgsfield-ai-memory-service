# Memory Service

A memory service for AI agents that ingests conversation turns, extracts structured knowledge via LLM, and answers recall queries with hybrid retrieval.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Service                        │
│                                                          │
│  POST /turns ──► Extraction Pipeline ──► Store           │
│       │              │                     │             │
│       │         DeepSeek V4 Flash     PostgreSQL         │
│       │         (via OpenRouter)      + pgvector         │
│       │                                    │             │
│  POST /recall ──► Hybrid Retrieve ──► Assemble Context  │
│                   │           │                          │
│              Vector Search  FTS Search                   │
│              (pgvector)    (tsvector)                    │
│                   │           │                          │
│                   └─── RRF ───┘                          │
│                        │                                 │
│               Noise Gate + Intent Matching               │
│                        │                                 │
│                  Tiered Assembly                          │
│              (facts → prefs → relevant → recent)         │
└─────────────────────────────────────────────────────────┘
```

The service is a single Python/FastAPI process backed by PostgreSQL with pgvector. All extraction, embedding, and retrieval happens synchronously within request handlers.

## Project Structure

```
src/
├── main.py              # entry point
├── ingestion/           # turn intake and memory extraction
│   ├── extraction.py    # typed Extraction seam + LLM adapter
│   ├── intake.py        # turn ingestion orchestration
│   └── memory_write.py  # Memory persistence and supersession writes
├── core/
│   ├── config.py        # settings (env vars)
│   ├── embeddings.py    # local embedding model
│   ├── lifecycle.py     # session/user delete + supersession repair
│   └── search.py        # /search endpoint logic
├── api/                 # HTTP layer
│   ├── app.py           # FastAPI app factory
│   ├── auth.py          # auth middleware
│   ├── bootstrap.py     # startup/lifespan
│   ├── routes.py        # endpoint handlers
│   └── schemas.py       # request/response models
├── recall/              # recall pipeline
│   ├── selection.py     # Recall orchestration
│   ├── planning.py      # noise gate, intent matching, expansion plan
│   ├── retrieval.py     # hybrid search
│   ├── ranking.py       # RRF fusion
│   └── budget.py        # tiered assembly
└── storage/             # persistence
    ├── database.py      # DB engine + session
    ├── models.py        # ORM models
    ├── rows.py          # typed query row interfaces
    └── store.py         # data access (fetches)

tests/
├── unit/                # pure logic (no network, no DB)
├── integration/         # tests hitting the live service
└── e2e/                 # multi-turn scenario tests
```

## Backing Store: PostgreSQL + pgvector

**Why Postgres:**
- Single store for relational data (memories with supersession chains), full-text search (tsvector/tsquery), and vector similarity (pgvector) — no multi-service coordination
- ACID transactions guarantee immediate read-after-write consistency
- Named Docker volume provides persistence across restarts
- Mature, battle-tested, easy to reason about

**Schema:**
- `turns` — raw conversation turns with JSONB messages, FTS index, and embedding (timezone-aware timestamps)
- `memories` — extracted structured knowledge with type, key, value, confidence, active flag, supersession pointers, and embedding

## Extraction Pipeline

When `POST /turns` is called:

1. **Concatenate** messages into a single text block
2. **Embed** the turn content locally (all-MiniLM-L6-v2, 384d)
3. **Fetch** existing active memories for the user (for contradiction detection)
4. **Call LLM** (DeepSeek V4 Flash via OpenRouter) with a structured extraction prompt that:
   - Extracts facts, preferences, opinions, and events
   - Detects implicit facts ("walking Biscuit" → has a pet)
   - Identifies contradictions against existing memories
   - Returns a normalized `key` for topic deduplication
5. **Validate** LLM output — malformed items (missing/non-text `value`, non-dict entries) are filtered out
6. **Embed** each extracted memory value
7. **Store** memories, handling supersession (mark old memory inactive, link via `supersedes`/`superseded_by`)

The Extraction seam accepts a typed request containing Turn text and Memory References.
Prompt construction, LLM transport, response cleanup, JSON parsing, validation, and known
failure handling sit behind that interface. Failures return an error-bearing result with
no candidate Memory records, so the Turn is still stored and Memory persistence is skipped.

**What we extract:**
- Personal facts (employment, location, family, pets)
- Preferences (communication style, tools, food)
- Opinions (with evolution tracking)
- Events (debugging sessions, interviews, moves)
- Corrections ("actually, I meant...")
- Implicit facts ("walking Biscuit this morning" → has a pet named Biscuit)

**What we might miss:**
- Very subtle implications requiring multi-turn reasoning
- Sarcasm or irony (LLM limitation)
- Facts stated in tool call outputs (handled but less tested)

## Recall Strategy

`POST /recall` uses a multi-stage pipeline:

### Stage 1: Hybrid Retrieve
- **Vector search**: Embed query → cosine similarity against memory embeddings (top-20)
- **Full-text search**: Postgres FTS with `plainto_tsquery` against memory values (top-20)
- Both searches are scoped to the user's active memories

### Stage 2: Rank with RRF
Reciprocal Rank Fusion combines both result sets: `score = Σ 1/(60 + rank_i)`

This handles both semantic queries ("tell me about their career") and keyword queries ("what's their dog's name?") well.

### Stage 3: Noise Gate + Intent Matching
Retrieved memories pass through a noise gate that prevents returning irrelevant results:
- **Vector similarity threshold** (≥0.45): semantically relevant memories pass regardless of keyword overlap
- **FTS score**: keyword matches pass
- **Token overlap**: direct word matches between query and memory
- **Intent matching**: query terms map to intent categories (employment, location, pet, diet, etc.) which match against memory keys

Special query types:
- **Profile queries** ("Tell me about this user"): return all stable facts without filtering
- **History queries** ("career history"): include superseded memories for the full arc

### Stage 4: Anchor Expansion
Once initial relevant memories are selected, the system expands to include related memories. If a query mentions "dog named Biscuit", the pet memory anchors expansion to also surface the user's location — enabling multi-hop recall.

### Stage 5: Tiered Assembly
Context is assembled in priority order, stopping when `max_tokens` is reached:

1. **Tier 1 — Facts** (employment, location, pets): ~50% of budget, always included first
2. **Tier 2 — Preferences** (communication style, dietary): ~25% of budget
3. **Tier 3 — Query-relevant** (opinions, events): remaining budget, ranked by RRF score
4. **Tier 4 — Recent turns** (last 5 from current session): conversational continuity, cut first

**Budget logic:** When `max_tokens` is tight, facts get priority positioning. Within each tier, items are capped (15 facts, 10 preferences, 10 relevant). Recent context is last priority and gets cut first.

Internally, Recall, Search, and Context assembly use typed `MemoryRow`,
`RecentTurnRow`, `RecallContext`, and `Citation` interfaces instead of raw
database row dictionaries or citation dictionaries. HTTP responses remain
plain JSON at the route seam.

## Fact Evolution

**Simple contradictions** (employment, location):
- LLM detects the contradiction against existing memories
- New memory is stored as `active=true`
- Old memory is set to `active=false` with `superseded_by` pointing to the new one
- `/recall` only surfaces active memories
- `/users/{user_id}/memories` shows the full history including superseded entries

**Session deletion and supersession integrity:**
- When a session is deleted, any memories that were superseded by memories in that session are reactivated
- If the deleted session is in the middle of a supersession chain, remaining memories are stitched together so newer active memories keep pointing to the nearest surviving ancestor
- This prevents "orphaned" inactive memories with dangling pointers

**Opinion arcs** (e.g., TypeScript opinions evolving):
- Each new opinion supersedes the previous one
- The latest opinion is what `/recall` returns
- The full arc is preserved in the supersession chain
- History queries can surface the full arc

**Corrections** ("actually, I meant..."):
- Treated the same as contradictions — the correction supersedes the incorrect fact

## Cross-Session Knowledge Sharing

Memories are stored at the **user level** and shared across sessions for the same `user_id`. Raw turn history is session-scoped.

This means:
- Facts extracted in session 1 are available when querying in session 3
- "Recent conversation context" in `/recall` only pulls from the current session
- Different `user_id`s never see each other's memories

## Tradeoffs

| Optimized for | Gave up |
|---|---|
| Extraction quality (LLM-based) | Latency on `/turns` (1-3s for LLM call) |
| Immediate consistency | Async throughput |
| Simplicity (single store) | Horizontal scalability |
| Local embeddings (no API key) | Embedding quality vs. OpenAI |
| Hybrid retrieval (vector + FTS) | Complexity vs. pure vector |
| Noise resistance (intent gate) | May miss edge-case semantic matches |

## Failure Modes

- **No data / cold session**: `/recall` returns `{"context": "", "citations": []}` — never errors
- **Missing LLM API key**: Extraction returns an error result, turn is still stored, memories won't be extracted
- **Malformed LLM output**: Items without text `value` are filtered out; service doesn't crash
- **Slow disk**: Queries may be slow but won't timeout (Postgres handles backpressure)
- **Malformed input**: FastAPI/Pydantic returns 422 with validation errors
- **Unicode/large payloads**: Handled gracefully — Postgres stores any valid UTF-8
- **Session delete with supersession**: Superseded memories are reactivated, no data loss

## Running

```bash
# Copy env and add your OpenRouter API key
cp .env.example .env
# Edit .env with your LLM_API_KEY

# Start the service
docker compose up -d

# Wait for health
until curl -sf http://localhost:8080/health; do sleep 1; done

# Run tests with the Docker commands in the next section
```

## Running Tests

Tests run against the live service:

```bash
# Rebuild and start the service
docker compose up -d --build service

# Copy local tests into the running service container
docker compose cp tests/. service:/app/tests

# Run all tests (unit + integration + e2e)
docker compose exec -T service python -m pytest tests -q

# Run specific test suites
docker compose exec -T service python -m pytest tests/unit -q
docker compose exec -T service python -m pytest tests/integration -q
docker compose exec -T service python -m pytest tests/e2e -q

# Run restart persistence test (requires docker access)
RUN_RESTART_TESTS=1 python3 -m unittest tests.integration.test_restart_persistence -v
```

Test coverage:
- **Contract**: roundtrip, empty session, search, memories endpoint, delete
- **Malformed input**: invalid JSON, missing fields, unicode, empty messages
- **Concurrent sessions**: user isolation
- **Recall quality**: fixture-based probe queries with expected facts
- **E2E**: fact evolution, cross-session knowledge, token budget, multi-hop, session delete, implicit facts, multi-message turns
- **Restart persistence**: data survives `docker compose restart`

## Code Quality

The codebase is formatted with `black` (line-length=100) and `isort` (black profile). Configuration is in `pyproject.toml`.

```bash
python3 -m black src/ tests/
python3 -m isort src/ tests/
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DB_HOST` | No | `db` | PostgreSQL host |
| `DB_PORT` | No | `5432` | PostgreSQL port |
| `DB_NAME` | No | `memory` | Database name |
| `DB_USER` | No | `memory` | Database user |
| `DB_PASSWORD` | No | `memory` | Database password |
| `DATABASE_URL` | No | _(built from above)_ | Full connection string override |
| `LLM_BASE_URL` | No | `https://openrouter.ai/api/v1` | OpenAI-compatible API endpoint |
| `LLM_API_KEY` | Yes | — | API key for the LLM provider |
| `LLM_MODEL` | No | `deepseek/deepseek-v4-flash` | Model identifier |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Local embedding model name |
| `MEMORY_AUTH_TOKEN` | No | — | Optional Bearer token for auth |

## License

MIT. See `LICENSE`.
