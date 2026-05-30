from __future__ import annotations

from dataclasses import dataclass

FACT_SHARE = 0.50
PREFERENCE_SHARE = 0.25
MAX_FACTS = 15
MAX_PREFERENCES = 10
MAX_RELEVANT = 10


@dataclass(frozen=True)
class ContextLine:
    text: str
    citation: dict


@dataclass
class TierResult:
    lines: list[str]
    citations: list[dict]
    spent: int


def assemble_context(
    memories: list[dict], recent_turns: list[dict], max_tokens: int
) -> tuple[str, list[dict]]:
    """Assemble Context with explicit tier budgets."""
    remaining = max_tokens
    sections: list[str] = []
    citations: list[dict] = []

    facts = [m for m in memories if m["type"] == "fact"]
    preferences = [m for m in memories if m["type"] == "preference"]
    others = [m for m in memories if m["type"] in ("opinion", "event")]

    fact_result = take_tier(
        memory_lines(facts[:MAX_FACTS], format_fact_line),
        cap=max(1, int(max_tokens * FACT_SHARE)),
        remaining=remaining,
        min_items=1,
    )
    remaining -= fact_result.spent
    if fact_result.lines:
        sections.append("## Known facts about this user\n" + "\n".join(fact_result.lines))
        citations.extend(fact_result.citations)

    pref_result = take_tier(
        memory_lines(preferences[:MAX_PREFERENCES], format_memory_line),
        cap=max(0, int(max_tokens * PREFERENCE_SHARE)),
        remaining=remaining,
    )
    remaining -= pref_result.spent
    if pref_result.lines:
        sections.append("## Preferences\n" + "\n".join(pref_result.lines))
        citations.extend(pref_result.citations)

    relevant_result = take_tier(
        memory_lines(others[:MAX_RELEVANT], format_relevant_line),
        cap=remaining,
        remaining=remaining,
    )
    remaining -= relevant_result.spent
    if relevant_result.lines:
        sections.append("## Relevant memories\n" + "\n".join(relevant_result.lines))
        citations.extend(relevant_result.citations)

    recent_result = take_tier(
        recent_lines(recent_turns),
        cap=remaining,
        remaining=remaining,
    )
    if recent_result.lines:
        sections.append("## Recent conversation context\n" + "\n".join(recent_result.lines))
        citations.extend(recent_result.citations)

    return "\n\n".join(sections), citations


def take_tier(
    lines: list[ContextLine],
    cap: int,
    remaining: int,
    min_items: int = 0,
) -> TierResult:
    selected: list[str] = []
    citations: list[dict] = []
    spent = 0

    for line in lines:
        cost = estimate_tokens(line.text)
        if cost > remaining - spent:
            break
        if spent + cost > cap and len(selected) >= min_items:
            break
        selected.append(line.text)
        citations.append(line.citation)
        spent += cost

    return TierResult(selected, citations, spent)


def memory_lines(memories: list[dict], formatter) -> list[ContextLine]:
    return [
        ContextLine(
            text=formatter(memory),
            citation={
                "turn_id": memory["source_turn_id"],
                "score": memory.get("rrf_score", 0),
                "snippet": memory["value"],
            },
        )
        for memory in memories
    ]


def recent_lines(recent_turns: list[dict]) -> list[ContextLine]:
    lines = []
    for turn in recent_turns:
        text = turn["content_text"][:200]
        lines.append(
            ContextLine(
                text=f"- [{str(turn.get('timestamp', ''))[:10]}] {text}",
                citation={
                    "turn_id": turn["id"],
                    "score": 0.0,
                    "snippet": text[:100],
                },
            )
        )
    return lines


def format_fact_line(memory: dict) -> str:
    if memory.get("active") is False:
        return f"- Previously: {memory['value']}"
    line = format_memory_line(memory)
    if memory.get("updated_at"):
        line += f" (updated {str(memory['updated_at'])[:10]})"
    return line


def format_memory_line(memory: dict) -> str:
    return f"- {memory['value']}"


def format_relevant_line(memory: dict) -> str:
    return f"- [{str(memory.get('created_at', ''))[:10]}] {memory['value']}"


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)
