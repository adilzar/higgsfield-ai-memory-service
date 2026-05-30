from datetime import datetime
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from src.database import async_session, init_db
from src.lifecycle import delete_session_data
from src.models import Memory, Turn

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_delete_latest_session_reactivates_previous_memory():
    await init_db()
    unique = uuid4().hex
    user_id = f"lifecycle-user-{unique}"
    old_session_id = f"session-old-{unique}"
    latest_session_id = f"session-latest-{unique}"
    old_turn_id = f"old-turn-{unique}"
    latest_turn_id = f"latest-turn-{unique}"
    old_id = f"old-{unique}"
    latest_id = f"latest-{unique}"

    async with async_session() as db:
        try:
            add_turn(db, old_session_id, user_id, old_turn_id)
            add_turn(db, latest_session_id, user_id, latest_turn_id)
            db.add(
                Memory(
                    id=old_id,
                    user_id=user_id,
                    session_id=old_session_id,
                    source_turn_id=old_turn_id,
                    type="fact",
                    key="location",
                    value="Lives in NYC",
                    active=False,
                    superseded_by=latest_id,
                )
            )
            db.add(
                Memory(
                    id=latest_id,
                    user_id=user_id,
                    session_id=latest_session_id,
                    source_turn_id=latest_turn_id,
                    type="fact",
                    key="location",
                    value="Lives in Berlin",
                    active=True,
                    supersedes=old_id,
                )
            )
            await db.commit()

            await delete_session_data(db, latest_session_id)

            old = await db.get(Memory, old_id)
            latest = await db.get(Memory, latest_id)

            assert latest is None
            assert old is not None
            assert old.active is True
            assert old.superseded_by is None
        finally:
            await delete_user_rows(db, user_id)


async def test_delete_middle_session_stitches_supersession_chain():
    await init_db()
    unique = uuid4().hex
    user_id = f"lifecycle-user-{unique}"
    old_session_id = f"session-old-{unique}"
    middle_session_id = f"session-middle-{unique}"
    current_session_id = f"session-current-{unique}"
    old_turn_id = f"old-turn-{unique}"
    middle_turn_id = f"middle-turn-{unique}"
    current_turn_id = f"current-turn-{unique}"
    old_id = f"old-{unique}"
    middle_id = f"middle-{unique}"
    current_id = f"current-{unique}"

    async with async_session() as db:
        try:
            add_turn(db, old_session_id, user_id, old_turn_id)
            add_turn(db, middle_session_id, user_id, middle_turn_id)
            add_turn(db, current_session_id, user_id, current_turn_id)
            db.add_all(
                [
                    Memory(
                        id=old_id,
                        user_id=user_id,
                        session_id=old_session_id,
                        source_turn_id=old_turn_id,
                        type="fact",
                        key="location",
                        value="Lives in NYC",
                        active=False,
                        superseded_by=middle_id,
                    ),
                    Memory(
                        id=middle_id,
                        user_id=user_id,
                        session_id=middle_session_id,
                        source_turn_id=middle_turn_id,
                        type="fact",
                        key="location",
                        value="Lives in Berlin",
                        active=False,
                        supersedes=old_id,
                        superseded_by=current_id,
                    ),
                    Memory(
                        id=current_id,
                        user_id=user_id,
                        session_id=current_session_id,
                        source_turn_id=current_turn_id,
                        type="fact",
                        key="location",
                        value="Lives in Tokyo",
                        active=True,
                        supersedes=middle_id,
                    ),
                ]
            )
            await db.commit()

            await delete_session_data(db, middle_session_id)

            old = await db.get(Memory, old_id)
            middle = await db.get(Memory, middle_id)
            current = await db.get(Memory, current_id)

            assert middle is None
            assert old is not None
            assert old.active is False
            assert old.superseded_by == current_id
            assert current is not None
            assert current.active is True
            assert current.supersedes == old_id
        finally:
            await delete_user_rows(db, user_id)


def add_turn(db, session_id: str, user_id: str, turn_id: str) -> None:
    db.add(
        Turn(
            id=turn_id,
            session_id=session_id,
            user_id=user_id,
            messages=[],
            timestamp=datetime.utcnow(),
            content_text="test turn",
        )
    )


async def delete_user_rows(db, user_id: str) -> None:
    memory_ids = select(Memory.id).where(Memory.user_id == user_id)
    await db.execute(delete(Memory).where(Memory.id.in_(memory_ids)))
    await db.execute(delete(Turn).where(Turn.user_id == user_id))
    await db.commit()
