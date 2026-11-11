"""Crash injection: a kill between the writes that file a reply must not cost a second payment."""

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from precis.llm import Gateway
from precis.llm.clock import ManualClock
from precis.llm.fake import FakeAnthropic
from precis.models import LLMCall
from precis.store import Store

from .helpers import config, echo, request


class SimulatedKill(BaseException):
    """Stands for SIGKILL: nothing after it runs, nothing catches it."""


class _KillOnCacheWrite:
    """A connection that dies when a response is written to the cache."""

    def __init__(self, db: Any) -> None:
        self._db = db

    def execute(self, sql: str, *args: Any) -> Any:
        if "INTO response_cache" in sql:
            raise SimulatedKill
        return self._db.execute(sql, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)


def _paid(calls: list[LLMCall]) -> Counter[str]:
    return Counter(c.request_hash for c in calls if c.status == "ok")


async def test_a_kill_while_filing_a_reply_does_not_pay_twice(tmp_path: Path) -> None:
    path = tmp_path / "precis.db"
    cfg = config(pipeline={"fallbacks": False})
    clock = ManualClock()
    fake = FakeAnthropic([echo], clock=clock)

    crashed = Store(path)
    crashed._db = _KillOnCacheWrite(crashed._db)  # type: ignore[assignment]
    gw = Gateway(fake, crashed, run_id="run-1", config=cfg, clock=clock)
    with pytest.raises(SimulatedKill):
        await gw.create(request("filed"), stage="read")

    with Store(path) as resumed:
        again = Gateway(fake, resumed, run_id="run-2", config=cfg, clock=clock)
        await again.create(request("filed"), stage="read")
        paid = _paid(resumed.calls())
        # Either the first reply was filed whole (served from the cache now)
        # or not at all (sent again, as a request in flight at a kill is):
        # never a paid row without its cache entry.
        assert max(paid.values()) == 1
        assert resumed.cache_get(next(iter(paid))) is not None


async def test_refusals_and_truncated_replies_are_recorded_without_a_cache_entry(
    store: Store, clock: ManualClock
) -> None:
    fake = FakeAnthropic([lambda p: echo(p)], clock=clock)
    gw = Gateway(fake, store, run_id="r", config=config(pipeline={"fallbacks": False}), clock=clock)
    reply = await gw.create(request("word " * 300, max_tokens=20), stage="read")
    assert reply.stop_reason == "max_tokens"
    [call] = store.calls()
    assert call.status == "ok"
    assert store.cache_get(call.request_hash) is None


class _KillOnBatchClose:
    """A connection that dies when a batch is marked collected."""

    def __init__(self, db: Any) -> None:
        self._db = db

    def execute(self, sql: str, *args: Any) -> Any:
        if sql.startswith("UPDATE batches SET collected_at"):
            raise SimulatedKill
        return self._db.execute(sql, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)


async def test_a_batch_collected_again_after_a_kill_is_filed_once(tmp_path: Path) -> None:
    path = tmp_path / "precis.db"
    cfg = config(limits={"batch_max_wait_s": 30, "batch_poll_s": 10})
    clock = ManualClock()
    fake = FakeAnthropic([echo], clock=clock, batch_latency_s=60)

    crashed = Store(path)
    crashed._db = _KillOnBatchClose(crashed._db)  # type: ignore[assignment]
    first = Gateway(fake, crashed, run_id="run-1", config=cfg, clock=clock, mode="batch")
    # The result is filed (and the caller served) before the kill.
    await first.create(request("collected twice"), stage="read")

    with Store(path) as resumed:
        assert len(resumed.open_batches()) == 1
        second = Gateway(fake, resumed, run_id="run-2", config=cfg, clock=clock, mode="batch")
        await second.create(request("starts the collection"), stage="read")
        for _ in range(1000):
            if not resumed.open_batches():
                break
            await asyncio.sleep(0)
        await second.aclose()
        assert resumed.open_batches() == []
        assert max(_paid(resumed.calls()).values()) == 1
