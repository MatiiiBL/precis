import asyncio
from datetime import UTC, datetime

import httpx2 as httpx
import pytest

from precis.config import FetchConfig
from precis.llm.clock import ManualClock
from precis.sources.http import ArxivHttp, FetchError, RateLimiter, parse_retry_after

URL = "https://export.arxiv.org/api/query"


def _client(
    handler: httpx.MockTransport | None,
    clock: ManualClock,
    **config: float | int | str,
) -> ArxivHttp:
    return ArxivHttp(
        FetchConfig.model_validate({"min_interval_s": 3.0, "max_attempts": 4, **config}),
        transport=handler,
        clock=clock,
        jitter=lambda: 1.0,  # backoff at its ceiling, so waits are exact
    )


def _scripted(statuses: list[int | Exception], seen: list[httpx.Request]) -> httpx.MockTransport:
    replies = iter(statuses)

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        headers = {"Retry-After": "7"} if reply == 429 else {}
        return httpx.Response(reply, headers=headers, text=f"status {reply}")

    return httpx.MockTransport(handle)


async def test_limiter_spaces_requests_from_any_number_of_tasks() -> None:
    clock = ManualClock()
    limiter = RateLimiter(3.0, clock)
    starts: list[float] = []

    async def request() -> None:
        await limiter.wait()
        starts.append(clock.now())

    await asyncio.gather(*(request() for _ in range(4)))
    assert starts == [0.0, 3.0, 6.0, 9.0]
    clock.advance(10)
    await request()  # after a pause, no wait at all
    assert starts[-1] == 19.0


async def test_hold_pushes_back_every_caller() -> None:
    clock = ManualClock()
    limiter = RateLimiter(3.0, clock)
    await limiter.wait()
    limiter.hold(30)
    await limiter.wait()
    assert clock.now() == 30.0


async def test_every_request_carries_the_user_agent() -> None:
    seen: list[httpx.Request] = []
    async with _client(_scripted([200], seen), ManualClock(), user_agent="precis-test/1") as http:
        response = await http.get(URL, params={"id_list": "2401.00001"})
    assert response.status_code == 200
    assert seen[0].headers["user-agent"] == "precis-test/1"
    assert seen[0].url.params["id_list"] == "2401.00001"


async def test_429_and_5xx_are_retried_with_backoff_and_retry_after() -> None:
    clock = ManualClock()
    seen: list[httpx.Request] = []
    async with _client(_scripted([429, 503, 200], seen), clock) as http:
        response = await http.get(URL)
    assert response.status_code == 200
    assert len(seen) == 3
    assert (http.requests, http.retries) == (3, 2)
    # First retry: Retry-After 7 s beats the 2 s backoff; second: 4 s backoff.
    assert clock.slept == [7.0, 4.0]


async def test_connection_errors_and_timeouts_are_retried() -> None:
    clock = ManualClock()
    seen: list[httpx.Request] = []
    errors: list[int | Exception] = [httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), 200]
    async with _client(_scripted(errors, seen), clock) as http:
        assert (await http.get(URL)).status_code == 200
    assert len(seen) == 3


async def test_retries_give_up_with_the_last_error() -> None:
    seen: list[httpx.Request] = []
    async with _client(_scripted([500] * 4, seen), ManualClock()) as http:
        with pytest.raises(FetchError, match=r"failed after 4 attempts: HTTP 500"):
            await http.get(URL)
    assert len(seen) == 4


async def test_other_statuses_are_returned_to_the_caller() -> None:
    seen: list[httpx.Request] = []
    async with _client(_scripted([404], seen), ManualClock()) as http:
        assert (await http.get(URL)).status_code == 404
    assert len(seen) == 1


def test_parse_retry_after() -> None:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert parse_retry_after("120") == 120.0
    assert parse_retry_after("Thu, 01 Jan 2026 12:00:30 GMT", now) == 30.0
    assert parse_retry_after("Thu, 01 Jan 2026 11:00:00 GMT", now) == 0.0
    assert parse_retry_after("soon") is None
    assert parse_retry_after(None) is None
