import json
import logging
from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

_client = None


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


def extract_memories(turn_content: str, existing_memories: list[dict]) -> list[dict]:
    """Call LLM to extract structured memories from a conversation turn."""
    existing_str = "None" if not existing_memories else json.dumps(
        [{"key": m["key"], "type": m["type"], "value": m["value"]} for m in existing_memories],
        indent=2
    )

    prompt = EXTRACTION_PROMPT.format(
        existing_memories=existing_str,
        turn_content=turn_content
    )

    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        return json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Extraction failed: {e}")
        return []
