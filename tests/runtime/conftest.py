import pytest

from precis.llm.clock import ManualClock
from precis.llm.fake import FakeAnthropic
from precis.store import Store

from .helpers import echo


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def fake(clock: ManualClock) -> FakeAnthropic:
    return FakeAnthropic([echo], clock=clock, seed=1)
