import pytest

from src.api.bootstrap import initialize_service


@pytest.mark.asyncio
async def test_initialize_service_initializes_db_then_warms_embedding():
    events = []

    async def db_init():
        events.append("db")

    def embed(text):
        events.append(f"embed:{text}")
        return [0.0]

    await initialize_service(db_init=db_init, embed=embed)

    assert events == ["db", "embed:warmup"]


@pytest.mark.asyncio
async def test_initialize_service_skips_warmup_when_db_init_fails():
    events = []

    async def db_init():
        events.append("db")
        raise RuntimeError("database unavailable")

    def embed(text):
        events.append(f"embed:{text}")
        return [0.0]

    with pytest.raises(RuntimeError, match="database unavailable"):
        await initialize_service(db_init=db_init, embed=embed)

    assert events == ["db"]
