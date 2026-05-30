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
│                  Tiered Assembly                          │
│              (facts → prefs → relevant → recent)         │
└─────────────────────────────────────────────────────────┘
```

The service is a single Python/FastAPI process backed by PostgreSQL with pgvector. All extraction, embedding, and retrieval happens synchronously within request handlers.

## Backing Store: PostgreSQL + pgvector

**Why Postgres:**
- Single store for relational data (memories with supersession chains), full-text search (tsvector/tsquery), and vector similarity (pgvector) — no multi-service coordination
- ACID transactions guarantee immediate read-after-write consistency
- Named Docker volume provides persistence across restarts
- Mature, battle-tested, easy to reason about

**Schema:**
- `turns` — raw conversation turns with JSONB messages, FTS index, and embedding
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
5. **Embed** each extracted memory value
6. **Store** memories, handling supersession (mark old memory inactive, link via `supersedes`/`superseded_by`)

**What we extract:**
- Personal facts (employment, location, family, pets)
- Preferences (communication style, tools, food)
- Opinions (with evolution tracking)
- Events (debugging sessions, interviews, moves)
- Corrections ("actually, I meant...")

**What we might miss:**
- Very subtle implications requiring multi-turn reasoning
- Sarcasm or irony (LLM limitation)
- Facts stated in tool call outputs (handled but less tested)

## Recall Strategy

`POST /recall` uses a three-stage pipeline:

### Stage 1: Hybrid Retrieve
- **Vector search**: Embed query → cosine similarity against memory embeddings (top-20)
- **Full-text search**: Postgres FTS with `plainto_tsquery` against memory values (top-20)
- Both searches are scoped to the user's active memories

### Stage 2: Rank with RRF
Reciprocal Rank Fusion combines both result sets: `score = Σ 1/(60 + rank_i)`

This handles both semantic queries ("tell me about their career") and keyword queries ("what's their dog's name?") well.

### Stage 3: Tiered Assembly
Context is assembled in priority order, stopping when `max_tokens` is reached:

1. **Tier 1 — Facts** (employment, location, pets): Always included first
2. **Tier 2 — Preferences** (communication style, dietary): High priority
3. **Tier 3 — Query-relevant** (opinions, events): Ranked by RRF score
4. **Tier 4 — Recent turns** (last 5 from current session): Conversational continuity

**Budget logic:** When `max_tokens` is tight, facts get ~50% of budget, then preferences, then relevant memories fill the rest. Recent context is last priority and gets cut first.

## Fact Evolution

**Simple contradictions** (employment, location):
- LLM detects the contradiction against existing memories
- New memory is stored as `active=true`
- Old memory is set to `active=false` with `superseded_by` pointing to the new one
- `/recall` only surfaces active memories
- `/users/{user_id}/memories` shows the full history including superseded entries

**Opinion arcs** (e.g., TypeScript opinions evolving):
- Each new opinion supersedes the previous one
- The latest opinion is what `/recall` returns
- The full arc is preserved in the supersession chain
- This is a simplification — a more sophisticated system could summarize the arc

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

## Failure Modes

- **No data / cold session**: `/recall` returns `{"context": "", "citations": []}` — never errors
- **Missing LLM API key**: Extraction returns empty list, turns are still stored (memories just won't be extracted)
- **Slow disk**: Queries may be slow but won't timeout (Postgres handles backpressure)
- **Malformed input**: FastAPI/Pydantic returns 422 with validation errors
- **Unicode/large payloads**: Handled gracefully — Postgres stores any valid UTF-8

## Running

```bash
# Copy env and add your OpenRouter API key
cp .env.example .env
# Edit .env with your LLM_API_KEY

# Start the service
docker compose up -d

# Wait for health
until curl -sf http://localhost:8080/health; do sleep 1; done

# Run tests (requires service running)
pip install httpx pytest
pytest tests/ -v
```

## Running Tests

Tests are integration tests that run against the live service:

```bash
# Start the service first
docker compose up -d

# Run all tests
pytest tests/ -v

# Run just the recall quality fixture
pytest tests/test_service.py::TestRecallQuality -v -s
```

The recall quality test ingests the fixture conversations, runs probe queries, and reports how many expected facts were found in the recall context.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_BASE_URL` | No | `https://openrouter.ai/api/v1` | OpenAI-compatible API endpoint |
| `LLM_API_KEY` | Yes | — | API key for the LLM provider |
| `LLM_MODEL` | No | `deepseek/deepseek-v4-flash` | Model identifier |
| `MEMORY_AUTH_TOKEN` | No | — | Optional Bearer token for auth |
