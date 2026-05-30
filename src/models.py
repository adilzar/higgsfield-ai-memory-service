import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Index, String, Text, text
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Turn(Base):
    __tablename__ = "turns"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=True, index=True)
    messages = Column(JSONB, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    metadata_ = Column("metadata", JSONB, nullable=True)
    content_text = Column(Text, nullable=False)  # concatenated for FTS
    embedding = Column(Vector(384), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    memories = relationship("Memory", back_populates="source_turn_rel")

    __table_args__ = (
        Index("ix_turns_fts", text("to_tsvector('english', content_text)"), postgresql_using="gin"),
    )


class Memory(Base):
    __tablename__ = "memories"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=True, index=True)
    session_id = Column(String, nullable=False, index=True)
    source_turn_id = Column(String, ForeignKey("turns.id", ondelete="CASCADE"), nullable=False)
    type = Column(String, nullable=False)  # fact, preference, opinion, event
    key = Column(String, nullable=False)  # normalized topic
    value = Column(Text, nullable=False)
    confidence = Column(Float, default=1.0)
    active = Column(Boolean, default=True, index=True)
    supersedes = Column(String, nullable=True)  # id of memory this replaces
    superseded_by = Column(String, nullable=True)  # id of memory that replaced this
    embedding = Column(Vector(384), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    source_turn_rel = relationship("Turn", back_populates="memories")

    __table_args__ = (
        Index("ix_memories_fts", text("to_tsvector('english', value)"), postgresql_using="gin"),
        Index("ix_memories_user_active", "user_id", "active"),
    )
