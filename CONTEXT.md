# Memory Service Context

Domain language for the Higgsfield memory service challenge. Use these terms when discussing module seams, tests, and README claims.

## Language

**Turn**:
A completed conversation exchange submitted to the service for storage and extraction. A Turn may contain user, assistant, and tool messages.
_Avoid_: Message batch, chat chunk

**Memory**:
Structured knowledge extracted from a Turn, with type, key, value, confidence, provenance, and active/supersession state.
_Avoid_: Raw chunk, note, embedding record

**Memory Lifecycle**:
Rules that preserve Memory active state and supersession links when Turns, Sessions, or Users are deleted.
_Avoid_: Cleanup, cascade logic

**Recall**:
The process that selects stored Memory and recent Turn evidence, then formats Context for the next agent turn.
_Avoid_: Retrieval, search, lookup

**Context**:
The formatted text returned by `/recall` and injected into the agent prompt.
_Avoid_: Results, payload, summary

**Citation**:
Provenance attached to Context, pointing back to the Turn that supports a surfaced Memory or recent excerpt.
_Avoid_: Metadata, reference

**Noise Gate**:
Recall rule that returns empty Context when a query asks about an unknown topic and does not match a broad profile intent.
_Avoid_: Relevance filter, threshold

**Anchor Expansion**:
Recall rule that widens evidence from a retrieved anchor Memory to related stable facts for the same user, enabling multi-hop Context without dumping the whole profile.
_Avoid_: Graph traversal, query expansion

**Deterministic Recall**:
Recall implementation that uses stored Memory types, keys, text matching, vector similarity, and full-text search without an LLM call on the Recall path.
_Avoid_: LLM rerank, judge pass
