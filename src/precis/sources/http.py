"""Polite HTTP for arXiv: one shared pace, a User-Agent, and retries with backoff.

arXiv asks automated clients to leave about three seconds between requests
and to back off when told to. Every request this package makes, to the API
and to arxiv.org downloads alike, goes through one ``ArxivHttp`` and its
``RateLimiter``, so concurrent downloads cannot add up to a faster pace. A
``Retry-After`` from the server pushes back the next slot for every caller,
not just the one that was refused.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from types import TracebackType

import httpx2 as httpx

from precis.config import FetchConfig
from precis.llm.clock import Clock, RealClock

logger = logging.getLogger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 60.0
# A Retry-After longer than this is treated as this long; the run should
# slow down, not stall for an hour on one header.
RETRY_AFTER_MAX_S = 300.0


class FetchError(Exception):
    """A request to arXiv failed for good (after retries, or with a fatal status)."""


class RateLimiter:
    """Starts requests at least ``min_interval_s`` apart, across every task sharing it.

    Waiters queue on a lock, so they are served in arrival order and each
    one's wait is measured from the previous request's start.
    """

    def __init__(self, min_interval_s: float, clock: Clock | None = None) -> None:
        self._interval = min_interval_s
        self._clock = clock or RealClock()
        self._next = float("-inf")
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        """Return when the caller may start its request."""
        async with self._lock:
            delay = self._next - self._clock.now()
            if delay > 0:
                await self._clock.sleep(delay)
            self._next = self._clock.now() + self._interval

    def hold(self, seconds: float) -> None:
        """Keep every caller waiting at least ``seconds`` from now (server back-off)."""
        self._next = max(self._next, self._clock.now() + seconds)


class ArxivHttp:
    """The one HTTP client for arxiv.org and export.arxiv.org.

    ``get`` returns the final response for any status it does not retry (the
    caller decides what a 404 means); 429, 5xx, timeouts and connection
    errors are retried with exponential backoff and full jitter, honouring
    ``Retry-After``, and raise ``FetchError`` when attempts run out.
    """

    def __init__(
        self,
        config: FetchConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Clock | None = None,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.limiter = RateLimiter(config.min_interval_s, clock)
        self._max_attempts = config.max_attempts
        self._jitter = jitter
        self._client = httpx.AsyncClient(
            headers={"User-Agent": config.user_agent},
            timeout=config.timeout_s,
            follow_redirects=True,
            transport=transport,
        )
        self.requests = 0
        self.retries = 0

    async def get(
        self, url: str, *, params: Mapping[str, str | int] | None = None
    ) -> httpx.Response:
        error = ""
        for attempt in range(1, self._max_attempts + 1):
            await self.limiter.wait()
            self.requests += 1
            retry_after: float | None = None
            try:
                response = await self._client.get(url, params=params)
            except httpx.TransportError as exc:
                error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code not in RETRY_STATUS:
                    return response
                error = f"HTTP {response.status_code}"
                retry_after = parse_retry_after(response.headers.get("retry-after"))
            if attempt == self._max_attempts:
                break
            delay = self._backoff(attempt, retry_after)
            self.retries += 1
            logger.info("GET %s: %s; retrying in %.1fs", url, error, delay)
            self.limiter.hold(delay)
        raise FetchError(f"GET {url} failed after {self._max_attempts} attempts: {error}")

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        ceiling = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2.0 ** (attempt - 1))
        delay = ceiling * self._jitter()
        if retry_after is not None:
            delay = max(delay, min(retry_after, RETRY_AFTER_MAX_S))
        return delay

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> ArxivHttp:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Seconds to wait from a ``Retry-After`` header (delta-seconds or HTTP date)."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())
