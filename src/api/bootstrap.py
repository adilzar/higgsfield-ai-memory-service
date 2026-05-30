from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from src.database import init_db
from src.embeddings import embed_text

logger = logging.getLogger(__name__)

WARMUP_TEXT = "warmup"


async def initialize_service(
    db_init: Callable[[], Awaitable[None]] = init_db,
    embed: Callable[[str], object] = embed_text,
    warmup_text: str = WARMUP_TEXT,
) -> None:
    await db_init()
    embed(warmup_text)
    logger.info("Memory service ready")
