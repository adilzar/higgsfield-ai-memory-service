from __future__ import annotations

import re
from dataclasses import dataclass

from src.storage.rows import MemoryRow

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
    "employment": {
        "work",
        "works",
        "job",
        "career",
        "employer",
        "company",
        "role",
        "pm",
        "engineer",
        "designer",
        "do",
        "occupation",
    },
    "pet": {"dog", "cat", "pet", "pets", "biscuit"},
    "diet": {
        "diet",
        "dietary",
        "vegetarian",
        "shellfish",
        "allergy",
        "allergic",
        "food",
        "eat",
    },
    "communication_style": {
        "concise",
        "direct",
        "answers",
        "style",
        "prefer",
        "preference",
        "fluff",
    },
    "opinion": {
        "think",
        "opinion",
        "programming",
        "language",
        "typescript",
        "python",
        "love",
    },
}

STABLE_INTENTS = {"location", "employment", "pet", "diet", "communication_style"}
HISTORY_TERMS = {"history", "previous", "previously", "past", "former", "before"}
VEC_SCORE_THRESHOLD = 0.45


@dataclass(frozen=True)
class QueryProfile:
    tokens: frozenset[str]
    intents: frozenset[str]
    is_profile_query: bool
    include_history: bool


@dataclass(frozen=True)
class RecallPlan:
    memories: list[MemoryRow]
    query: QueryProfile
    expansion_intents: frozenset[str]


def build_recall_plan(
    query: str, retrieved: list[MemoryRow], scope_memories: list[MemoryRow]
) -> RecallPlan:
    """Plan deterministic Recall policy for a query and candidate Memory rows."""
    query_profile = _profile_query(query)
    selected: list[MemoryRow] = []
    selected_ids: set[str] = set()

    def add(memory: MemoryRow) -> None:
        mid = memory.id
        if mid not in selected_ids:
            selected.append(memory)
            selected_ids.add(mid)

    if query_profile.is_profile_query:
        for memory in scope_memories:
            if memory.active is True and _memory_intents(memory) & STABLE_INTENTS:
                add(memory)
        return RecallPlan(
            memories=selected,
            query=query_profile,
            expansion_intents=frozenset(),
        )

    relevant = [
        memory
        for memory in retrieved
        if _memory_matches_query(memory, set(query_profile.tokens), set(query_profile.intents))
    ]

    for memory in relevant:
        add(memory)

    if query_profile.intents:
        for memory in scope_memories:
            if memory.active is False:
                continue
            if _memory_intents(memory) & query_profile.intents:
                add(memory)

    if not selected:
        return RecallPlan(
            memories=[],
            query=query_profile,
            expansion_intents=frozenset(),
        )

    expansion_intents = set(query_profile.intents)
    for memory in selected:
        expansion_intents.update(_memory_intents(memory))

    if query_profile.include_history:
        expansion_intents.update({"employment", "location", "opinion"})

    for memory in scope_memories:
        if memory.active is False and not query_profile.include_history:
            continue
        if _memory_intents(memory) & expansion_intents:
            add(memory)

    return RecallPlan(
        memories=selected,
        query=query_profile,
        expansion_intents=frozenset(expansion_intents),
    )


def select_recall_memories(
    query: str, retrieved: list[MemoryRow], scope_memories: list[MemoryRow]
) -> list[MemoryRow]:
    return build_recall_plan(query, retrieved, scope_memories).memories


def _profile_query(query: str) -> QueryProfile:
    query_tokens = _tokens(query)
    query_intents = _intents_for_query(query)
    return QueryProfile(
        tokens=frozenset(query_tokens),
        intents=frozenset(query_intents),
        is_profile_query=_is_profile_query(query, query_intents),
        include_history=needs_history(query),
    )


def _memory_matches_query(
    memory: MemoryRow, query_tokens: set[str], query_intents: set[str]
) -> bool:
    if memory.vec_score and memory.vec_score >= VEC_SCORE_THRESHOLD:
        return True

    if memory.fts_score and memory.fts_score > 0:
        return True

    memory_tokens = _tokens(
        " ".join(
            [
                memory.key,
                memory.type,
                memory.value,
            ]
        )
    )
    if query_tokens & memory_tokens:
        return True

    return bool(query_intents & _memory_intents(memory))


def _intents_for_query(query: str) -> set[str]:
    query_tokens = _tokens(query)
    query_intents = {intent for intent, terms in INTENT_TERMS.items() if query_tokens & terms}
    if "preference" in query_intents:
        query_intents.add("communication_style")
    return query_intents


def _memory_intents(memory: MemoryRow) -> set[str]:
    key = memory.key.lower()
    memory_type = memory.type.lower()
    intents: set[str] = set()

    if any(part in key for part in ("location", "city", "home", "moved")):
        intents.add("location")
    if any(
        part in key
        for part in (
            "employment",
            "career",
            "job",
            "work",
            "company",
            "occupation",
            "role",
        )
    ):
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
    raw_query_tokens = _raw_tokens(query)
    if "profile" in raw_query_tokens:
        return True
    if (
        {"remember", "know"} & raw_query_tokens
        and {"me", "user"} & raw_query_tokens
        and not query_intents
    ):
        return True
    if "everything" in raw_query_tokens and {"me", "user"} & raw_query_tokens:
        return True
    if (
        {"tell", "about"} <= raw_query_tokens
        and {"me", "user"} & raw_query_tokens
        and not query_intents
    ):
        return True
    return False


def needs_history(query: str) -> bool:
    return bool(_tokens(query) & HISTORY_TERMS)


def _tokens(text: str) -> set[str]:
    return {token for token in _raw_tokens(text) if token not in STOPWORDS and len(token) > 1}


def _raw_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))
