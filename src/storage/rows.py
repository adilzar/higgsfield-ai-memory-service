from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class MemoryRow:
    id: str
    type: str
    key: str
    value: str
    confidence: float | None
    session_id: str
    source_turn_id: str
    created_at: Any = None
    updated_at: Any = None
    active: bool | None = True
    supersedes: str | None = None
    superseded_by: str | None = None
    vec_score: float | None = None
    fts_score: float | None = None
    rrf_score: float = 0.0

    @classmethod
    def from_mapping(cls, row: Any) -> "MemoryRow":
        data = dict(row)
        return cls(
            id=data["id"],
            type=data["type"],
            key=data["key"],
            value=data["value"],
            confidence=data.get("confidence"),
            session_id=data["session_id"],
            source_turn_id=data["source_turn_id"],
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            active=data.get("active", True),
            supersedes=data.get("supersedes"),
            superseded_by=data.get("superseded_by"),
            vec_score=_optional_float(data.get("vec_score")),
            fts_score=_optional_float(data.get("fts_score")),
            rrf_score=float(data.get("rrf_score") or 0.0),
        )

    def with_scores(
        self,
        *,
        vec_score: float | None = None,
        fts_score: float | None = None,
        rrf_score: float | None = None,
    ) -> "MemoryRow":
        return replace(
            self,
            vec_score=self.vec_score if vec_score is None else vec_score,
            fts_score=self.fts_score if fts_score is None else fts_score,
            rrf_score=self.rrf_score if rrf_score is None else rrf_score,
        )


@dataclass(frozen=True)
class RecentTurnRow:
    id: str
    content_text: str
    timestamp: Any = None

    @classmethod
    def from_mapping(cls, row: Any) -> "RecentTurnRow":
        data = dict(row)
        return cls(
            id=data["id"],
            content_text=data["content_text"],
            timestamp=data.get("timestamp"),
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
