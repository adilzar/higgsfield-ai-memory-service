import pytest

from src.lifecycle import delete_session_data
from src.models import Memory
from tests.db_helpers import db_scenario

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_delete_latest_session_reactivates_previous_memory():
    async with db_scenario("lifecycle") as (db, scenario):
        scenario.add_turn(db, "old")
        scenario.add_turn(db, "latest")
        latest_id = scenario.memory_id("latest")
        old_id = scenario.add_memory(
            db,
            "old",
            value="Lives in NYC",
            active=False,
            superseded_by=latest_id,
        )
        scenario.add_memory(
            db,
            "latest",
            value="Lives in Berlin",
            active=True,
            supersedes=old_id,
        )
        await db.commit()

        await delete_session_data(db, scenario.session_id("latest"))

        old = await db.get(Memory, old_id)
        latest = await db.get(Memory, latest_id)

        assert latest is None
        assert old is not None
        assert old.active is True
        assert old.superseded_by is None


async def test_delete_middle_session_stitches_supersession_chain():
    async with db_scenario("lifecycle") as (db, scenario):
        scenario.add_turn(db, "old")
        scenario.add_turn(db, "middle")
        scenario.add_turn(db, "current")
        middle_id = scenario.memory_id("middle")
        current_id = scenario.memory_id("current")
        old_id = scenario.add_memory(
            db,
            "old",
            value="Lives in NYC",
            active=False,
            superseded_by=middle_id,
        )
        scenario.add_memory(
            db,
            "middle",
            value="Lives in Berlin",
            active=False,
            supersedes=old_id,
            superseded_by=current_id,
        )
        scenario.add_memory(
            db,
            "current",
            value="Lives in Tokyo",
            active=True,
            supersedes=middle_id,
        )
        await db.commit()

        await delete_session_data(db, scenario.session_id("middle"))

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
