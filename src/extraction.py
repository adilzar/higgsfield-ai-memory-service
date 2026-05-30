import json
import logging
from dataclasses import dataclass

from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

_client = None
MEMORY_TYPES = {"fact", "preference", "opinion", "event"}


def get_llm_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    return _client


EXTRACTION_PROMPT = """You are a memory extraction system. Analyze the conversation turn and extract structured memories.

EXISTING MEMORIES for this user (use these to detect contradictions/updates):
{existing_memories}

CONVERSATION TURN:
{turn_content}

Extract memories as a JSON array. Each memory object must have:
- "type": one of "fact", "preference", "opinion", "event"
- "key": normalized topic slug (e.g., "employment", "location", "pet", "food_preference", "communication_style")
- "value": concise statement of the memory (e.g., "Works at Notion as a PM")
- "confidence": float 0.0-1.0 (1.0 for explicit statements, 0.5-0.8 for implicit/inferred)
- "supersedes_key": if this memory contradicts/updates an existing memory, put the key of the old memory here. null otherwise.

Rules:
- Extract BOTH explicit facts ("I work at Notion") and implicit ones ("walking Biscuit this morning" → has a pet named Biscuit)
- Detect corrections: "actually, I meant..." or "sorry, not X — Y" should supersede the old fact
- For opinions that evolve (not simple contradictions), create a new memory with the updated opinion but still mark it as superseding the old one
- Do NOT extract trivial conversational filler
- Do NOT extract facts about the assistant, only about the user
- If nothing meaningful to extract, return an empty array []

Respond with ONLY a JSON array, no other text."""


class ExtractionError(Exception):
    """Raised when LLM extraction fails in a non-recoverable way."""

    pass


@dataclass
class ExtractedMemory:
    type: str
    key: str
    value: str
    confidence: float = 1.0
    supersedes_key: str | None = None


@dataclass
class ExtractionResult:
    memories: list[ExtractedMemory]
    raw_response: str | None = None


def _build_prompt(turn_content: str, existing_memories: list[dict]) -> str:
    existing_str = (
        "None"
        if not existing_memories
        else json.dumps(
            [{"key": m["key"], "type": m["type"], "value": m["value"]} for m in existing_memories],
            indent=2,
        )
    )
    return EXTRACTION_PROMPT.format(existing_memories=existing_str, turn_content=turn_content)


def _call_llm(prompt: str) -> str:
    """Call the LLM and return raw response content. Raises ExtractionError on API failure."""
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        content = response.choices[0].message.content
        if not content:
            raise ExtractionError("LLM returned empty content")
        return content.strip()
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"LLM API call failed: {e}") from e


def parse_extraction_response(raw: str) -> list[ExtractedMemory]:
    """Parse LLM response into validated memories. Raises ExtractionError on malformed output."""
    content = raw
    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"LLM returned invalid JSON: {e}\nRaw: {content[:200]}") from e

    if not isinstance(parsed, list):
        raise ExtractionError(f"Expected JSON array, got {type(parsed).__name__}")

    return [memory for item in parsed if (memory := _coerce_memory(item)) is not None]


def _coerce_memory(item: object) -> ExtractedMemory | None:
    if not isinstance(item, dict):
        return None

    value = item.get("value")
    if not isinstance(value, str) or not value.strip():
        return None

    memory_type = _coerce_text(item.get("type"), default="fact")
    if memory_type not in MEMORY_TYPES:
        memory_type = "fact"

    supersedes_key = item.get("supersedes_key")
    if not isinstance(supersedes_key, str) or not supersedes_key.strip():
        supersedes_key = None

    return ExtractedMemory(
        type=memory_type,
        key=_coerce_text(item.get("key"), default="unknown"),
        value=value.strip(),
        confidence=_coerce_confidence(item.get("confidence")),
        supersedes_key=supersedes_key.strip() if supersedes_key else None,
    )


def _coerce_text(value: object, *, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _coerce_confidence(value: object) -> float:
    if isinstance(value, bool):
        return 1.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, confidence))


def extract_memories(turn_content: str, existing_memories: list[dict]) -> ExtractionResult:
    """Extract structured memories from a conversation turn.

    Returns ExtractionResult with memories list (possibly empty).
    Raises ExtractionError on failures that the caller should handle.
    """
    prompt = _build_prompt(turn_content, existing_memories)
    raw = _call_llm(prompt)
    memories = parse_extraction_response(raw)
    return ExtractionResult(memories=memories, raw_response=raw)
