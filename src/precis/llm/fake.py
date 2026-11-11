"""``FakeAnthropic``: an in-process stand-in for the parts of the SDK client we use.

It implements ``messages.create``, ``messages.with_raw_response.create``,
the same pair under ``beta.messages`` (server-side fallbacks), and
``messages.batches`` (``create``, ``retrieve``, ``results``), and it returns
the SDK's own ``Message`` / ``BetaMessage`` / ``MessageBatch`` objects and
raises the SDK's own exception classes with real ``httpx2`` responses and
headers. Code under test therefore takes exactly the paths it takes against
the real API, including header parsing and error classification.

What the fake replies is decided by *responders*: functions from the request
``params`` to a ``FakeReply`` or ``None``; the first non-``None`` reply wins.
Around them the fake models what matters for the runtime:

- usage that behaves like the API's: input tokens from
  ``precis.util.estimate_tokens``, prompt caching as a prefix match with
  breakpoints, TTLs and a minimum cacheable length, output tokens from the
  reply's content (capped by ``max_tokens``);
- latency (``LatencyModel``) on an injectable clock;
- server-side rate limits (``ServerLimits``): token buckets for requests,
  input and output tokens per minute that answer 429 with ``retry-after``
  and send ``anthropic-ratelimit-*`` headers on every response;
- faults (``Faults``): random or scripted 429, 529, 500, timeouts and 400s,
  overload above a concurrency threshold, and errored or expired batch items;
- the documented 400s for request shapes current models reject (sampling
  parameters, thinking budgets, forced tool choice).

All randomness comes from one ``random.Random(seed)``, so a run is
reproducible given the same seed and the same order of requests.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any, Literal, NoReturn

import anthropic
import httpx2
from anthropic.types import Message, StopReason
from anthropic.types.beta import BetaMessage
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

from precis.llm.clock import Clock, RealClock
from precis.llm.limiter import TokenBucket
from precis.llm.models import FALLBACK_BETA, FALLBACK_TARGETS
from precis.llm.pricing import PRICES
from precis.util import canonical_json, estimate_tokens, sha256_hex

type Responder = Callable[[dict[str, Any]], FakeReply | None]
type FaultKind = Literal[
    "rate_limit",
    "overloaded",
    "server_error",
    "timeout",
    "bad_request",
    "stream_error",  # an overloaded_error event after a streamed response started
    "stream_timeout",  # the stream stops sending data after it started
]

# Fixed origin for the timestamps the fake reports (batch times, reset
# headers), so they depend only on the clock and not on the wall date.
EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

_STREAM_FAULTS = ("stream_error", "stream_timeout")

# The API routes ``fallbacks="default"`` by refusal category; the fake always
# takes a model's first listed target, and never retries these categories.
_NOT_RETRIED_CATEGORIES = {"reasoning_extraction"}

# Models that reject forced tool choice and explicit thinking configuration
# other than adaptive.
_STRICT_MODELS = {
    "claude-fable-5-2",
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
}

_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


@dataclass
class FakeReply:
    """One scripted reply.

    Either a complete ``message`` (its ``usage`` is kept when non-zero and
    estimated otherwise; ``id`` and ``model`` are taken as given), or the
    lighter form: ``content`` as a string or a list of content-block dicts
    plus ``stop_reason``. ``model`` overrides the model reported in the
    reply (by default the requested one).
    """

    content: str | list[dict[str, Any]] = ""
    stop_reason: StopReason = "end_turn"
    message: Message | None = None
    model: str | None = None
    stop_details: dict[str, Any] | None = None
    output_tokens: int | None = None

    @classmethod
    def text(cls, text: str) -> FakeReply:
        return cls(content=[{"type": "text", "text": text}])

    @classmethod
    def tool_use(cls, name: str, tool_input: dict[str, Any], *, text: str = "") -> FakeReply:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        blocks.append({"type": "tool_use", "name": name, "input": tool_input})
        return cls(content=blocks, stop_reason="tool_use")

    @classmethod
    def refusal(cls, category: str | None = "cyber") -> FakeReply:
        return cls(
            content=[],
            stop_reason="refusal",
            stop_details={"type": "refusal", "category": category, "explanation": None},
        )


@dataclass(frozen=True)
class LatencyModel:
    """Time to first token plus generation time, with log-normal jitter."""

    first_token_s: float = 0.8
    seconds_per_output_token: float = 0.01
    jitter: float = 0.25
    error_s: float = 0.1

    def sample(self, rng: random.Random, output_tokens: int) -> float:
        mean = self.first_token_s + output_tokens * self.seconds_per_output_token
        if self.jitter <= 0:
            return mean
        return mean * rng.lognormvariate(-(self.jitter**2) / 2, self.jitter)


@dataclass(frozen=True)
class ServerLimits:
    """Per-minute limits the fake enforces like the API: continuously refilled buckets.

    ``burst_s`` is how many seconds of tokens a bucket holds; the API's
    documented behaviour is a full minute, and a smaller value simulates
    its stricter enforcement over short intervals.
    """

    rpm: int
    itpm: int
    otpm: int
    burst_s: float = 60.0


@dataclass(frozen=True)
class Faults:
    """Random fault injection; probabilities are per request (or per batch item)."""

    rate_limit: float = 0.0
    overloaded: float = 0.0
    server_error: float = 0.0
    timeout: float = 0.0
    retry_after_s: float = 5.0
    timeout_s: float = 30.0
    overload_above: int | None = None
    """Answer 529 to requests arriving while more than this many are in flight."""
    batch_errored: float = 0.0
    batch_expired: float = 0.0


@dataclass(frozen=True)
class FakeEvent:
    """One request as the fake saw it, for measuring achieved rates."""

    t: float
    outcome: Literal["ok", "429", "529", "500", "timeout", "400"]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    in_flight: int = 0


@dataclass
class _Batch:
    id: str
    created_at: float
    ends_at: float
    requests: list[tuple[str, dict[str, Any]]]
    results: list[dict[str, Any]] | None = None


@dataclass
class _Segment:
    tokens: int
    digest: str
    breakpoint_ttl: Literal["5m", "1h"] | None


@dataclass
class _CacheUsage:
    input_tokens: int
    cache_read: int
    cache_write: int
    cache_write_1h: int


class FakeAnthropic:
    """A deterministic, in-process ``AsyncAnthropic`` for tests and offline runs."""

    def __init__(
        self,
        responders: Sequence[Responder] = (),
        *,
        clock: Clock | None = None,
        seed: int = 0,
        latency: LatencyModel | None = None,
        faults: Faults | None = None,
        limits: ServerLimits | None = None,
        batch_latency_s: float = 300.0,
        cache_min_tokens: int = 512,
    ) -> None:
        self.responders = list(responders)
        self.clock: Clock = clock or RealClock()
        self.rng = random.Random(seed)
        self.latency = latency or LatencyModel(first_token_s=0.0, seconds_per_output_token=0.0)
        self.faults = faults or Faults()
        self.batch_latency_s = batch_latency_s
        self.cache_min_tokens = cache_min_tokens
        self.events: list[FakeEvent] = []
        self.requests: list[dict[str, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.streamed = 0
        self._scripted: list[FaultKind] = []
        self._limits: ServerLimits | None = None
        self._buckets: tuple[TokenBucket, TokenBucket, TokenBucket] | None = None
        self._cache: dict[str, float] = {}
        self._batches: dict[str, _Batch] = {}
        self._counter = 0
        self.set_limits(limits)
        self.messages = _Messages(self)
        self.beta = _Beta(self)

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    def add_responder(self, responder: Responder) -> None:
        self.responders.append(responder)

    def fail_next(self, *kinds: FaultKind) -> None:
        """Make the next requests fail with these faults, in order."""
        self._scripted.extend(kinds)

    def set_limits(self, limits: ServerLimits | None) -> None:
        """Change the simulated server limits; the buckets start full."""
        self._limits = limits
        if limits is None:
            self._buckets = None
            return
        self._buckets = (
            TokenBucket(self.clock, limits.rpm, burst_s=limits.burst_s),
            TokenBucket(self.clock, limits.itpm, burst_s=limits.burst_s),
            TokenBucket(self.clock, limits.otpm, burst_s=limits.burst_s),
        )

    def count(self, outcome: str) -> int:
        """How many requests ended with ``outcome`` ("ok", "429", "529", ...)."""
        return sum(e.outcome == outcome for e in self.events)

    @property
    def batch_sizes(self) -> list[int]:
        """Number of requests in each batch created so far, in creation order."""
        return [len(b.requests) for b in self._batches.values()]

    # ------------------------------------------------------------------
    # the request path
    # ------------------------------------------------------------------

    async def _create(self, params: dict[str, Any], *, beta: bool, stream: bool = False) -> _Served:
        """Serve one request: the message as JSON, its sequence number and any stream fault.

        Faults of the ``stream_*`` kinds happen after the response has
        started; a non-streaming request gets the equivalent error at once.
        """
        self._counter += 1
        n = self._counter
        model = str(params.get("model", ""))
        self.requests.append(params)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            self._validate(params, beta=beta)
            fault = self._pick_fault()
            if fault is not None and fault not in _STREAM_FAULTS:
                await self._inject(fault, model)
            usage = self._prompt_usage(params)
            self._admit(model, usage.input_tokens + usage.cache_write)
            body = self._respond(params, usage, beta=beta, n=n)
            output_tokens = int(body["usage"]["output_tokens"])
            await self.clock.sleep(self.latency.sample(self.rng, output_tokens))
            if self._buckets is not None:
                self._buckets[2].take(output_tokens)
            outcome: Literal["ok", "529", "timeout"] = "ok"
            if fault is not None:
                outcome = "timeout" if fault == "stream_timeout" else "529"
            self.events.append(
                FakeEvent(
                    self.clock.now(),
                    outcome,
                    model,
                    usage.input_tokens + usage.cache_write,
                    output_tokens,
                    self.in_flight,
                )
            )
            if fault is not None and not stream:
                self._raise_stream_fault(fault, self._response_headers(n))
            return _Served(body, n, fault)
        finally:
            self.in_flight -= 1

    def _raise_stream_fault(self, fault: FaultKind, headers: httpx2.Headers) -> NoReturn:
        """Fail a response that has started, the way the SDK reports it."""
        request = httpx2.Request("POST", _MESSAGES_URL)
        if fault == "stream_timeout":
            raise anthropic.APITimeoutError(request=request)
        # An ``error`` event inside a 200 stream: the SDK raises the base
        # APIStatusError with the event's error type.
        body = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        response = httpx2.Response(200, headers=headers, json=body, request=request)
        raise anthropic.APIStatusError(
            f"Error event mid-stream: {body}", response=response, body=body
        )

    def _validate(self, params: dict[str, Any], *, beta: bool) -> None:
        missing = [k for k in ("model", "max_tokens", "messages") if k not in params]
        if missing:
            self._raise(400, "invalid_request_error", f"{missing[0]}: Field required")
        model = str(params["model"])
        if model not in PRICES:
            self._raise(404, "not_found_error", f"model: {model}")
        for name in ("temperature", "top_p", "top_k"):
            if name in params:
                self._raise(400, "invalid_request_error", f"{name}: not supported on this model")
        thinking = params.get("thinking") or {}
        if thinking.get("type") == "enabled" or "budget_tokens" in thinking:
            self._raise(400, "invalid_request_error", '"thinking.type.enabled" is not supported')
        if model in _STRICT_MODELS:
            if thinking.get("type") == "disabled":
                self._raise(400, "invalid_request_error", '"thinking.type.disabled" not supported')
            if (params.get("tool_choice") or {}).get("type") in ("any", "tool"):
                self._raise(
                    400,
                    "invalid_request_error",
                    'tool_choice: type "tool" and "any" are not supported for this model.',
                )
        if "fallbacks" in params and not beta:
            self._raise(400, "invalid_request_error", "fallbacks: needs the beta endpoint")

    def _pick_fault(self) -> FaultKind | None:
        """The fault for the next request: scripted first, then by chance."""
        if self._scripted:
            return self._scripted.pop(0)
        f = self.faults
        if f.overload_above is not None and self.in_flight > f.overload_above:
            return "overloaded"
        draw = self.rng.random()
        odds: tuple[tuple[FaultKind, float], ...] = (
            ("rate_limit", f.rate_limit),
            ("overloaded", f.overloaded),
            ("server_error", f.server_error),
            ("timeout", f.timeout),
        )
        for kind, p in odds:
            if draw < p:
                return kind
            draw -= p
        return None

    async def _inject(self, kind: FaultKind, model: str) -> NoReturn:
        """Fail the request before any response, as the API would."""
        if kind == "timeout":
            await self.clock.sleep(self.faults.timeout_s)
            self.events.append(FakeEvent(self.clock.now(), "timeout", model))
            raise anthropic.APITimeoutError(request=httpx2.Request("POST", _MESSAGES_URL))
        await self.clock.sleep(self.latency.error_s)
        if kind == "rate_limit":
            self.events.append(FakeEvent(self.clock.now(), "429", model))
            self._raise(
                429,
                "rate_limit_error",
                "This request would exceed your rate limit.",
                {"retry-after": _seconds(self.faults.retry_after_s), **self._limit_headers()},
            )
        if kind == "overloaded":
            self.events.append(FakeEvent(self.clock.now(), "529", model))
            self._raise(529, "overloaded_error", "Overloaded")
        if kind == "server_error":
            self.events.append(FakeEvent(self.clock.now(), "500", model))
            self._raise(500, "api_error", "Internal server error")
        self.events.append(FakeEvent(self.clock.now(), "400", model))
        self._raise(400, "invalid_request_error", "Injected invalid request")

    def _admit(self, model: str, counted_input: int) -> None:
        """Charge the request and its input to the buckets, or answer 429."""
        if self._buckets is None:
            return
        requests, inputs, outputs = self._buckets
        wait = max(
            requests.wait_time(1),
            inputs.wait_time(counted_input),
            # Output is metered as it is generated: a request may start while
            # the output bucket is not in debt.
            0.0 if outputs.level > 0 else outputs.wait_time(1),
        )
        if wait > 0:
            self.events.append(FakeEvent(self.clock.now(), "429", model, counted_input))
            self._raise(
                429,
                "rate_limit_error",
                "This request would exceed your rate limit.",
                {"retry-after": _seconds(wait), **self._limit_headers()},
            )
        requests.take(1)
        inputs.take(counted_input)

    def _respond(
        self, params: dict[str, Any], usage: _CacheUsage, *, beta: bool, n: int
    ) -> dict[str, Any]:
        requested = str(params["model"])
        reply = self._reply(params)
        body = self._message_body(reply, params, usage, n)
        fallbacks = params.get("fallbacks")
        category = (body.get("stop_details") or {}).get("category")
        if (
            beta
            and fallbacks
            and body["stop_reason"] == "refusal"
            and category not in _NOT_RETRIED_CATEGORIES
        ):
            first: dict[str, Any] = {} if fallbacks == "default" else next(iter(fallbacks), {})
            target = (
                FALLBACK_TARGETS.get(requested, (None,))[0]
                if fallbacks == "default"
                else first.get("model")
            )
            if target is not None:
                body = self._fallback_body(params, usage, body, target, n)
        return body

    def _fallback_body(
        self,
        params: dict[str, Any],
        usage: _CacheUsage,
        declined: dict[str, Any],
        target: str,
        n: int,
    ) -> dict[str, Any]:
        reply = self._reply({**params, "model": target})
        served = self._message_body(reply, {**params, "model": target}, usage, n)
        served["model"] = target
        served["content"] = [
            {
                "type": "fallback",
                "from": {"model": params["model"]},
                "to": {"model": target},
                "trigger": {"type": "refusal", "category": declined["stop_details"]["category"]},
            },
            *served["content"],
        ]
        u = served["usage"]
        served["usage"]["iterations"] = [
            {**_iteration(declined["usage"]), "type": "message", "model": params["model"]},
            {**_iteration(u), "type": "fallback_message", "model": target},
        ]
        return served

    def _reply(self, params: dict[str, Any]) -> FakeReply:
        for responder in self.responders:
            reply = responder(params)
            if reply is not None:
                return reply
        last = params.get("messages", [{}])[-1]
        raise LookupError(f"no responder matched request ending in {canonical_json(last)[:200]}")

    def _message_body(
        self, reply: FakeReply, params: dict[str, Any], usage: _CacheUsage, n: int
    ) -> dict[str, Any]:
        if reply.message is not None:
            body = reply.message.model_dump(mode="json", by_alias=True, exclude_none=True)
            given = body.get("usage") or {}
            if any(given.get(k) for k in ("input_tokens", "output_tokens")):
                return body
            output_tokens = reply.output_tokens or _output_tokens(body["content"])
            body["usage"] = _usage_json(usage, output_tokens)
            return body
        content = (
            [{"type": "text", "text": reply.content}]
            if isinstance(reply.content, str)
            else [dict(b) for b in reply.content]
        )
        for i, block in enumerate(content):
            if block.get("type") == "tool_use" and "id" not in block:
                block["id"] = f"toolu_fake_{n:06d}_{i}"
        output_tokens = reply.output_tokens or _output_tokens(content)
        stop_reason: str = reply.stop_reason
        max_tokens = int(params["max_tokens"])
        if output_tokens > max_tokens:
            output_tokens, stop_reason = max_tokens, "max_tokens"
        return {
            "id": f"msg_fake_{n:06d}",
            "type": "message",
            "role": "assistant",
            "model": reply.model or params["model"],
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": reply.stop_details,
            "usage": _usage_json(usage, output_tokens),
        }

    # ------------------------------------------------------------------
    # prompt caching
    # ------------------------------------------------------------------

    def _prompt_usage(self, params: dict[str, Any]) -> _CacheUsage:
        """Split the prompt into uncached input, cache reads and cache writes.

        Caching is a prefix match: the prompt is a sequence of blocks
        (tools, system, then message content), each prefix has a digest,
        reads hit the longest prefix written before and still alive, and a
        request writes the prefix up to its last breakpoint (an explicit
        ``cache_control`` or, with top-level ``cache_control``, the last
        block) when that prefix is at least ``cache_min_tokens`` long.
        """
        segments = _segments(params)
        prefix = list(itertools.accumulate(s.tokens for s in segments))
        total = prefix[-1] if prefix else 0
        breakpoints = [i for i, s in enumerate(segments) if s.breakpoint_ttl is not None]
        if not breakpoints:
            return _CacheUsage(total, 0, 0, 0)
        now = self.clock.now()
        hit = max(
            (i for i in range(breakpoints[-1] + 1) if self._cache.get(segments[i].digest, 0) > now),
            default=-1,
        )
        read = prefix[hit] if hit >= 0 else 0
        if hit >= 0:
            # A read refreshes the entry it hit.
            self._cache[segments[hit].digest] = max(self._cache[segments[hit].digest], now + 300)
        write = write_1h = 0
        covered = read
        for i in breakpoints:
            if i <= hit or prefix[i] < self.cache_min_tokens:
                continue
            new = prefix[i] - covered
            covered = prefix[i]
            write += new
            if segments[i].breakpoint_ttl == "1h":
                write_1h += new
                self._cache[segments[i].digest] = now + 3600
            else:
                self._cache[segments[i].digest] = now + 300
        return _CacheUsage(total - read - write, read, write, write_1h)

    # ------------------------------------------------------------------
    # headers and errors
    # ------------------------------------------------------------------

    def _limit_headers(self) -> dict[str, str]:
        if self._buckets is None or self._limits is None:
            return {}
        headers: dict[str, str] = {}
        now = self.clock.now()
        limits = (self._limits.rpm, self._limits.itpm, self._limits.otpm)
        names = ("requests", "input-tokens", "output-tokens")
        for name, limit, bucket in zip(names, limits, self._buckets, strict=True):
            level = max(0.0, bucket.level)
            remaining = math.floor(level) if name == "requests" else round(level, -3)
            to_full = (bucket.capacity - level) * 60.0 / limit
            reset = EPOCH + timedelta(seconds=now + to_full)
            headers[f"anthropic-ratelimit-{name}-limit"] = str(limit)
            headers[f"anthropic-ratelimit-{name}-remaining"] = str(int(remaining))
            headers[f"anthropic-ratelimit-{name}-reset"] = reset.isoformat().replace("+00:00", "Z")
        return headers

    def _response_headers(self, n: int) -> httpx2.Headers:
        return httpx2.Headers(
            {"request-id": f"req_fake_{n:06d}", "content-type": "application/json"}
            | self._limit_headers()
        )

    def _raise(
        self,
        status: int,
        error_type: str,
        message: str,
        headers: Mapping[str, str] | None = None,
    ) -> NoReturn:
        self._counter += 1
        request_id = f"req_fake_{self._counter:06d}"
        body = {
            "type": "error",
            "error": {"type": error_type, "message": message},
            "request_id": request_id,
        }
        response = httpx2.Response(
            status,
            headers={"request-id": request_id, **(headers or {})},
            json=body,
            request=httpx2.Request("POST", _MESSAGES_URL),
        )
        error_class = _ERRORS.get(status, anthropic.APIStatusError)
        raise error_class(f"Error code: {status} - {body}", response=response, body=body)

    # ------------------------------------------------------------------
    # batches
    # ------------------------------------------------------------------

    def _create_batch(self, requests: Iterable[Any]) -> MessageBatch:
        self._counter += 1
        now = self.clock.now()
        items = [(str(r["custom_id"]), dict(r["params"])) for r in requests]
        if not items:
            self._raise(400, "invalid_request_error", "requests: at least one request")
        if len({cid for cid, _ in items}) != len(items):
            self._raise(400, "invalid_request_error", "requests: custom_id values must be unique")
        for _, params in items:
            if "fallbacks" in params or params.get("stream"):
                self._raise(400, "invalid_request_error", "batch requests can't set this field")
        batch = _Batch(f"msgbatch_fake_{self._counter:06d}", now, now + self.batch_latency_s, items)
        self._batches[batch.id] = batch
        return self._batch_object(batch)

    def _batch_object(self, batch: _Batch) -> MessageBatch:
        now = self.clock.now()
        ended = now >= batch.ends_at
        if ended and batch.results is None:
            batch.results = [self._batch_result(cid, params) for cid, params in batch.requests]
        counts = dict.fromkeys(("processing", "succeeded", "errored", "canceled", "expired"), 0)
        if batch.results is None:
            counts["processing"] = len(batch.requests)
        else:
            for result in batch.results:
                counts[result["result"]["type"]] += 1

        def stamp(t: float) -> str:
            return (EPOCH + timedelta(seconds=t)).isoformat()

        return MessageBatch.model_validate(
            {
                "id": batch.id,
                "type": "message_batch",
                "processing_status": "ended" if ended else "in_progress",
                "request_counts": counts,
                "created_at": stamp(batch.created_at),
                "expires_at": stamp(batch.created_at + 86400),
                "ended_at": stamp(batch.ends_at) if ended else None,
                "archived_at": None,
                "cancel_initiated_at": None,
                "results_url": f"https://api.anthropic.com/v1/messages/batches/{batch.id}/results"
                if ended
                else None,
            }
        )

    def _batch_result(self, custom_id: str, params: dict[str, Any]) -> dict[str, Any]:
        draw = self.rng.random()
        if draw < self.faults.batch_expired:
            return {"custom_id": custom_id, "result": {"type": "expired"}}
        if draw < self.faults.batch_expired + self.faults.batch_errored:
            return _errored(custom_id, "api_error", "Internal server error")
        self._counter += 1
        try:
            self._validate(params, beta=False)
            body = self._respond(params, self._prompt_usage(params), beta=False, n=self._counter)
        except anthropic.APIStatusError as e:
            return _errored(custom_id, "invalid_request_error", str(e.api_message))
        model = str(params["model"])
        self.events.append(
            FakeEvent(
                self.clock.now(),
                "ok",
                model,
                body["usage"]["input_tokens"],
                body["usage"]["output_tokens"],
            )
        )
        return {"custom_id": custom_id, "result": {"type": "succeeded", "message": body}}

    def _batch(self, message_batch_id: str) -> _Batch:
        batch = self._batches.get(message_batch_id)
        if batch is None:
            self._raise(404, "not_found_error", f"batch {message_batch_id} not found")
        return batch

    async def _results(
        self, message_batch_id: str
    ) -> AsyncIterator[MessageBatchIndividualResponse]:
        batch = self._batch(message_batch_id)
        if batch.results is None:
            self._raise(400, "invalid_request_error", "batch has not ended yet")
        for result in batch.results or []:
            yield MessageBatchIndividualResponse.model_validate(result)
            await asyncio.sleep(0)


_ERRORS: dict[int, type[anthropic.APIStatusError]] = {
    400: anthropic.BadRequestError,
    404: anthropic.NotFoundError,
    429: anthropic.RateLimitError,
    500: anthropic.InternalServerError,
    529: anthropic.OverloadedError,
}


class FakeRawResponse[T]:
    """What ``with_raw_response.create`` returns: headers now, the message on ``parse()``."""

    def __init__(self, headers: httpx2.Headers, message: T) -> None:
        self.headers = headers
        self._message = message

    async def parse(self) -> T:
        return self._message


@dataclass(frozen=True)
class _Served:
    body: dict[str, Any]
    n: int
    fault: FaultKind | None


@dataclass(frozen=True)
class _StreamResponse:
    headers: httpx2.Headers


class FakeMessageStream[T]:
    """What ``async with messages.stream(...) as stream`` yields."""

    def __init__(
        self, fake: FakeAnthropic, headers: httpx2.Headers, message: T, fault: FaultKind | None
    ) -> None:
        self.response = _StreamResponse(headers)
        self._fake = fake
        self._message = message
        self._fault = fault

    async def get_final_message(self) -> T:
        if self._fault is not None:
            self._fake._raise_stream_fault(self._fault, self.response.headers)
        return self._message


class FakeMessageStreamManager[T]:
    """``messages.stream(...)``: the request is sent when the block is entered."""

    def __init__(self, open_stream: Callable[[], Awaitable[FakeMessageStream[T]]]) -> None:
        self._open = open_stream

    async def __aenter__(self) -> FakeMessageStream[T]:
        return await self._open()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
        /,
    ) -> None:
        return None


class _RawMessages:
    def __init__(self, fake: FakeAnthropic) -> None:
        self._fake = fake

    async def create(self, **params: Any) -> FakeRawResponse[Message]:
        served = await self._fake._create(params, beta=False)
        return FakeRawResponse(
            self._fake._response_headers(served.n), Message.model_validate(served.body)
        )


class _RawBetaMessages:
    def __init__(self, fake: FakeAnthropic) -> None:
        self._fake = fake

    async def create(self, **params: Any) -> FakeRawResponse[BetaMessage]:
        served = await self._fake._create(_beta_params(self._fake, params), beta=True)
        return FakeRawResponse(
            self._fake._response_headers(served.n), BetaMessage.model_validate(served.body)
        )


class _Batches:
    def __init__(self, fake: FakeAnthropic) -> None:
        self._fake = fake

    async def create(self, *, requests: Iterable[Any]) -> MessageBatch:
        return self._fake._create_batch(requests)

    async def retrieve(self, message_batch_id: str) -> MessageBatch:
        return self._fake._batch_object(self._fake._batch(message_batch_id))

    async def results(self, message_batch_id: str) -> AsyncIterator[MessageBatchIndividualResponse]:
        self._fake._batch(message_batch_id)
        return self._fake._results(message_batch_id)


class _Messages:
    def __init__(self, fake: FakeAnthropic) -> None:
        self._fake = fake
        self.with_raw_response = _RawMessages(fake)
        self.batches = _Batches(fake)

    async def create(self, **params: Any) -> Message:
        raw = await self.with_raw_response.create(**params)
        return await raw.parse()

    def stream(self, **params: Any) -> FakeMessageStreamManager[Message]:
        async def open_stream() -> FakeMessageStream[Message]:
            self._fake.streamed += 1
            served = await self._fake._create(params, beta=False, stream=True)
            message = Message.model_validate(served.body)
            headers = self._fake._response_headers(served.n)
            return FakeMessageStream(self._fake, headers, message, served.fault)

        return FakeMessageStreamManager(open_stream)


class _BetaMessages:
    def __init__(self, fake: FakeAnthropic) -> None:
        self._fake = fake
        self.with_raw_response = _RawBetaMessages(fake)

    async def create(self, **params: Any) -> BetaMessage:
        raw = await self.with_raw_response.create(**params)
        return await raw.parse()

    def stream(self, **params: Any) -> FakeMessageStreamManager[BetaMessage]:
        async def open_stream() -> FakeMessageStream[BetaMessage]:
            self._fake.streamed += 1
            sent = _beta_params(self._fake, params)
            served = await self._fake._create(sent, beta=True, stream=True)
            message = BetaMessage.model_validate(served.body)
            headers = self._fake._response_headers(served.n)
            return FakeMessageStream(self._fake, headers, message, served.fault)

        return FakeMessageStreamManager(open_stream)


def _beta_params(fake: FakeAnthropic, params: dict[str, Any]) -> dict[str, Any]:
    """The request body of a beta call: ``betas`` travels as a header, not in the body."""
    sent = dict(params)
    betas = sent.pop("betas", [])
    if "fallbacks" in sent and FALLBACK_BETA not in betas:
        fake._raise(400, "invalid_request_error", "fallbacks: requires the beta header")
    return sent


class _Beta:
    def __init__(self, fake: FakeAnthropic) -> None:
        self.messages = _BetaMessages(fake)


def _seconds(value: float) -> str:
    return str(max(1, math.ceil(value)))


def _errored(custom_id: str, error_type: str, message: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "result": {
            "type": "errored",
            "error": {"type": "error", "error": {"type": error_type, "message": message}},
        },
    }


def _usage_json(usage: _CacheUsage, output_tokens: int) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": usage.cache_write,
        "cache_read_input_tokens": usage.cache_read,
        "cache_creation": {
            "ephemeral_5m_input_tokens": usage.cache_write - usage.cache_write_1h,
            "ephemeral_1h_input_tokens": usage.cache_write_1h,
        },
    }


def _iteration(usage: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    return {k: usage[k] for k in keys}


def _block_tokens(block: Any) -> int:
    if isinstance(block, str):
        return estimate_tokens(block)
    if isinstance(block, dict):
        for key in ("text", "thinking"):
            if isinstance(block.get(key), str):
                return estimate_tokens(block[key])
        stripped = {k: v for k, v in block.items() if k != "cache_control"}
        return estimate_tokens(canonical_json(stripped))
    return estimate_tokens(canonical_json(block))


def _output_tokens(content: list[dict[str, Any]]) -> int:
    return max(1, sum(_block_tokens(b) for b in content))


def _segments(params: dict[str, Any]) -> list[_Segment]:
    """The prompt as cacheable blocks in render order: tools, system, messages."""
    blocks: list[Any] = list(params.get("tools") or [])
    system = params.get("system")
    if isinstance(system, str):
        blocks.append({"type": "text", "text": system})
    elif system:
        blocks.extend(system)
    for message in params.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content, "role": message.get("role")})
        else:
            blocks.extend(content or [])
    segments: list[_Segment] = []
    digest = sha256_hex(str(params.get("model", "")))
    for block in blocks:
        control = block.get("cache_control") if isinstance(block, dict) else None
        stripped = (
            {k: v for k, v in block.items() if k != "cache_control"}
            if isinstance(block, dict)
            else block
        )
        digest = sha256_hex(digest + canonical_json(stripped))
        ttl: Literal["5m", "1h"] | None = None
        if control:
            ttl = "1h" if control.get("ttl") == "1h" else "5m"
        segments.append(_Segment(_block_tokens(block), digest, ttl))
    auto = params.get("cache_control")
    if auto and segments and segments[-1].breakpoint_ttl is None:
        segments[-1].breakpoint_ttl = "1h" if auto.get("ttl") == "1h" else "5m"
    return segments
