"""``Gateway``: the one door every model request goes through.

``create`` takes the keyword arguments of ``client.messages.create`` and
adds what no caller should have to think about: the response cache, the
budget, admission control (``precis.llm.limiter``), retries with backoff,
server-side refusal fallbacks, and an ``LLMCall`` row with the cost of every
response. In batch mode the same call is served through the Message
Batches API at half price; see ``precis.llm.batch``.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import anthropic
from anthropic.types import Message
from anthropic.types.beta import BetaMessage

from precis.config import Config
from precis.llm.base import Reply
from precis.llm.batch import BatchQueue
from precis.llm.client import AsyncAnthropicLike
from precis.llm.clock import Clock, RealClock
from precis.llm.errors import BudgetExceeded, GatewayError, RefusalError
from precis.llm.limiter import Admission, Outcome, Ticket, Usage
from precis.llm.models import FALLBACK_BETA, FALLBACK_TARGETS, supports_fallbacks
from precis.llm.pricing import check_priced, cost, estimate_cost
from precis.models import LLMCall
from precis.store import Store
from precis.util import canonical_json, estimate_tokens, sha256_hex

log = logging.getLogger(__name__)

# Starting guesses until a stage has real responses to average over.
_DEFAULT_OUTPUT_TOKENS = 1000
_EWMA_WEIGHT = 0.2


@dataclass(frozen=True)
class GatewayStats:
    """A snapshot of the gateway's counters, for progress views and reports."""

    in_flight: int
    queued: int
    """Requests waiting for admission (a concurrency slot or bucket capacity)."""
    concurrency_limit: int
    calls: int
    """Completed API requests (responses received, refusals included)."""
    cached_calls: int
    retries: int
    rate_limited: int
    """429 responses."""
    overloaded: int
    """529 responses."""
    errors: int
    """Requests that failed after retries or with a non-retryable error."""
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cost_usd: float
    batches_pending: int
    """Batches submitted and not yet ended (0 in online mode)."""
    refusals: int = 0
    server_errors: int = 0
    timeouts: int = 0
    batch_calls: int = 0
    rpm_limit: float | None = None
    itpm_limit: float | None = None
    otpm_limit: float | None = None


@dataclass
class _Counters:
    calls: int = 0
    cached_calls: int = 0
    retries: int = 0
    rate_limited: int = 0
    overloaded: int = 0
    server_errors: int = 0
    timeouts: int = 0
    errors: int = 0
    refusals: int = 0
    batch_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class _Estimate:
    """What a request is expected to use, for reservations and the budget.

    ``prompt_tokens`` is the whole prompt by ``estimate_tokens``;
    ``input_tokens`` is the part expected to count against the input-token
    limit (the prompt scaled by the stage's observed ratio, which is below 1
    when most of it is read from the prompt cache).
    """

    prompt_tokens: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class _Response:
    reply: Reply
    attempts: int
    latency_s: float
    request_id: str | None
    batch: bool = False


class Gateway:
    """Cache, admission, retries, fallbacks, budget and accounting for model requests."""

    def __init__(
        self,
        client: AsyncAnthropicLike,
        store: Store,
        *,
        run_id: str,
        config: Config,
        clock: Clock | None = None,
        mode: Literal["online", "batch"] = "online",
    ) -> None:
        """Check that every configured model is priced, then set up admission control.

        Raises ``precis.llm.pricing.UnpricedModelError`` for a model (or one
        of its fallback targets) with no price: running it would record
        calls as free.
        """
        self._client = client
        self._store = store
        self._run_id = run_id
        self._config = config
        self._clock: Clock = clock or RealClock()
        self._mode: Literal["online", "batch"] = mode
        self._rng = random.Random(sha256_hex(run_id))
        check_priced(self._models_in_use())
        limits = config.limits
        self._admission = Admission(
            self._clock,
            initial_concurrency=limits.initial_concurrency,
            max_concurrency=limits.max_concurrency,
            rpm=limits.rpm,
            itpm=limits.itpm,
            otpm=limits.otpm,
            burst_s=limits.burst_s,
        )
        self._batches = (
            BatchQueue(client, self._clock, limits, self._rng) if mode == "batch" else None
        )
        self._counters = _Counters()
        self._reserved_usd = 0.0
        self._output_avg: dict[str, float] = {}
        self._input_ratio: dict[str, float] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def mode(self) -> Literal["online", "batch"]:
        return self._mode

    @property
    def spent_usd(self) -> float:
        return self._counters.cost_usd

    def stats(self) -> GatewayStats:
        c = self._counters
        batches = self._batches
        return GatewayStats(
            in_flight=self._admission.in_flight + (batches.in_batches if batches else 0),
            queued=self._admission.queued + (batches.queued if batches else 0),
            concurrency_limit=self._admission.concurrency_limit,
            calls=c.calls,
            cached_calls=c.cached_calls,
            retries=c.retries,
            rate_limited=c.rate_limited,
            overloaded=c.overloaded,
            errors=c.errors,
            input_tokens=c.input_tokens,
            output_tokens=c.output_tokens,
            cache_creation_input_tokens=c.cache_creation_input_tokens,
            cache_read_input_tokens=c.cache_read_input_tokens,
            cost_usd=c.cost_usd,
            batches_pending=batches.batches_pending if batches else 0,
            refusals=c.refusals,
            server_errors=c.server_errors,
            timeouts=c.timeouts,
            batch_calls=c.batch_calls,
            rpm_limit=self._admission.requests.rate_per_min,
            itpm_limit=self._admission.input_tokens.rate_per_min,
            otpm_limit=self._admission.output_tokens.rate_per_min,
        )

    async def aclose(self) -> None:
        """Send any queued batch requests and wait for every pending batch to finish."""
        if self._batches is not None:
            await self._batches.aclose()

    async def create(
        self, params: dict[str, Any], *, stage: str, paper_id: str | None = None
    ) -> Reply:
        """Send one Messages API request (or answer it from the cache).

        Raises ``RefusalError`` when the final reply is a refusal,
        ``BudgetExceeded`` before a request that would cross the budget, and
        ``GatewayError`` when the request fails for good.
        """
        model = str(params.get("model", ""))
        check_priced([model])
        request_hash = sha256_hex(canonical_json(params))
        started_at = datetime.now(UTC)
        if self._config.pipeline.use_cache:
            cached = self._store.cache_get(request_hash)
            if cached is not None:
                return self._serve_cached(cached, request_hash, stage, paper_id, started_at)

        estimate = self._estimate(params, stage)
        self._check_budget(estimate)
        self._reserved_usd += estimate.cost_usd
        try:
            if self._batches is None:
                response = await self._send_online(params, estimate)
            else:
                result = await self._batches.submit(params)
                response = _Response(
                    result.message, result.attempts, result.latency_s, None, batch=True
                )
        except GatewayError as e:
            self._counters.errors += 1
            self._record_failure(model, request_hash, stage, paper_id, started_at, e)
            raise
        finally:
            self._reserved_usd -= estimate.cost_usd

        self._learn(stage, estimate, response.reply)
        return self._accept(response, request_hash, stage, paper_id, started_at)

    # ------------------------------------------------------------------
    # cache, budget and estimates
    # ------------------------------------------------------------------

    def _serve_cached(
        self,
        cached: dict[str, Any],
        request_hash: str,
        stage: str,
        paper_id: str | None,
        started_at: datetime,
    ) -> Reply:
        reply: Reply
        if cached.get("kind") == "beta":
            reply = BetaMessage.model_validate(cached["message"])
        else:
            reply = Message.model_validate(cached["message"])
        self._counters.cached_calls += 1
        self._store.record_call(
            LLMCall(
                call_id=uuid.uuid4().hex,
                run_id=self._run_id,
                paper_id=paper_id,
                stage=stage,
                model=reply.model,
                request_hash=request_hash,
                status="cached",
                stop_reason=reply.stop_reason,
                started_at=started_at,
            )
        )
        return reply

    def _estimate(self, params: dict[str, Any], stage: str) -> _Estimate:
        prompt = {k: params.get(k) for k in ("system", "messages", "tools")}
        prompt_tokens = max(1, estimate_tokens(canonical_json(prompt)))
        ratio = min(1.0, self._input_ratio.get(stage, 1.0))
        input_tokens = max(1, round(prompt_tokens * ratio))
        max_tokens = int(params.get("max_tokens", _DEFAULT_OUTPUT_TOKENS))
        output_tokens = max(
            1, min(max_tokens, round(self._output_avg.get(stage, _DEFAULT_OUTPUT_TOKENS)))
        )
        usd = estimate_cost(
            str(params["model"]),
            input_tokens,
            output_tokens,
            cache_read_tokens=prompt_tokens - input_tokens,
            batch=self._mode == "batch",
        )
        return _Estimate(prompt_tokens, input_tokens, output_tokens, usd)

    def _check_budget(self, estimate: _Estimate) -> None:
        budget = self._config.budget.max_cost_usd
        if budget is None:
            return
        projected = self._counters.cost_usd + self._reserved_usd + estimate.cost_usd
        if projected > budget:
            raise BudgetExceeded(
                f"spent ${self._counters.cost_usd:.4f} (+${self._reserved_usd:.4f} in flight); "
                f"this request (~${estimate.cost_usd:.4f}) would cross the "
                f"${budget:.2f} budget"
            )

    def _learn(self, stage: str, estimate: _Estimate, reply: Reply) -> None:
        """Update the per-stage output average and the counted-input ratio.

        The ratio is measured against the unscaled prompt estimate, never
        against a reservation that already applied an earlier ratio: with
        many requests in flight that would feed the ratio back into itself.
        """
        usage = reply.usage
        counted = usage.input_tokens + (usage.cache_creation_input_tokens or 0)
        ratio = counted / estimate.prompt_tokens
        self._input_ratio[stage] = _ewma(self._input_ratio.get(stage), ratio)
        self._output_avg[stage] = _ewma(self._output_avg.get(stage), usage.output_tokens)

    # ------------------------------------------------------------------
    # online requests
    # ------------------------------------------------------------------

    def _request_kwargs(self, params: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """The arguments to send, and whether they go through the beta endpoint."""
        if self._config.pipeline.fallbacks and supports_fallbacks(str(params["model"])):
            return {**params, "betas": [FALLBACK_BETA], "fallbacks": "default"}, True
        return params, False

    async def _send_online(self, params: dict[str, Any], estimate: _Estimate) -> _Response:
        kwargs, beta = self._request_kwargs(params)
        limits = self._config.limits
        last_error: Exception | None = None
        for attempt in range(1, limits.max_attempts + 1):
            ticket = await self._admission.acquire(estimate.input_tokens, estimate.output_tokens)
            sent_at = self._clock.now()
            try:
                reply, headers = await self._call(kwargs, beta)
            except anthropic.APIStatusError as e:
                delay = self._on_status_error(e, ticket, attempt)
                last_error = e
            except anthropic.APIConnectionError as e:
                self._admission.release(ticket, Outcome.FAILURE)
                if isinstance(e, anthropic.APITimeoutError):
                    self._counters.timeouts += 1
                delay = self._backoff(attempt)
                last_error = e
            except BaseException:
                self._admission.release(ticket, Outcome.FAILURE)
                raise
            else:
                usage = reply.usage
                counted = Usage(
                    usage.input_tokens + (usage.cache_creation_input_tokens or 0),
                    usage.output_tokens,
                )
                self._admission.release(ticket, Outcome.SUCCESS, counted)
                self._admission.observe(headers)
                return _Response(
                    reply,
                    attempts=attempt,
                    latency_s=self._clock.now() - sent_at,
                    request_id=headers.get("request-id"),
                )
            if attempt == limits.max_attempts:
                break
            self._counters.retries += 1
            log.debug("attempt %d failed (%s); retrying in %.1fs", attempt, last_error, delay)
            await self._clock.sleep(delay)
        raise GatewayError(
            f"request failed after {limits.max_attempts} attempts: {last_error}",
            attempts=limits.max_attempts,
        ) from last_error

    async def _call(self, kwargs: dict[str, Any], beta: bool) -> tuple[Reply, dict[str, str]]:
        reply: Reply
        if beta:
            raw_beta = await self._client.beta.messages.with_raw_response.create(**kwargs)
            reply = await raw_beta.parse()
            headers = dict(raw_beta.headers)
        else:
            raw = await self._client.messages.with_raw_response.create(**kwargs)
            reply = await raw.parse()
            headers = dict(raw.headers)
        return reply, headers

    def _on_status_error(
        self, error: anthropic.APIStatusError, ticket: Ticket, attempt: int
    ) -> float:
        """Classify a status error: release the slot and return the retry delay, or raise."""
        headers = error.response.headers
        self._admission.observe(headers)
        if isinstance(error, anthropic.EnforcedSpendLimitReachedError):
            # The monthly spend cap: retrying fails until access resumes.
            self._admission.release(ticket, Outcome.FAILURE)
            raise GatewayError(f"spend limit reached: {error}", attempts=attempt) from error
        if isinstance(error, anthropic.RateLimitError):
            self._counters.rate_limited += 1
            self._admission.release(ticket, Outcome.OVERLOAD)
            retry_after = _retry_after(headers.get("retry-after"))
            if retry_after is not None:
                self._admission.pause(retry_after)
                return retry_after + self._rng.uniform(0, self._config.limits.backoff_base_s)
            return self._backoff(attempt)
        if isinstance(error, anthropic.OverloadedError):
            self._counters.overloaded += 1
            self._admission.release(ticket, Outcome.OVERLOAD)
            return self._backoff(attempt)
        if isinstance(error, anthropic.InternalServerError):
            self._counters.server_errors += 1
            self._admission.release(ticket, Outcome.FAILURE)
            return self._backoff(attempt)
        self._admission.release(ticket, Outcome.FAILURE)
        raise GatewayError(f"request rejected: {error}", attempts=attempt) from error

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter: uniform in [0, base * 2^(attempt-1)]."""
        limits = self._config.limits
        ceiling = min(limits.backoff_max_s, limits.backoff_base_s * 2 ** (attempt - 1))
        return self._rng.uniform(0, ceiling)

    # ------------------------------------------------------------------
    # accounting
    # ------------------------------------------------------------------

    def _accept(
        self,
        response: _Response,
        request_hash: str,
        stage: str,
        paper_id: str | None,
        started_at: datetime,
    ) -> Reply:
        reply = response.reply
        usage = reply.usage
        usd = cost(usage, reply.model, batch=response.batch)
        c = self._counters
        c.calls += 1
        c.batch_calls += response.batch
        c.input_tokens += usage.input_tokens
        c.output_tokens += usage.output_tokens
        c.cache_creation_input_tokens += usage.cache_creation_input_tokens or 0
        c.cache_read_input_tokens += usage.cache_read_input_tokens or 0
        c.cost_usd += usd
        refused = reply.stop_reason == "refusal"
        self._store.record_call(
            LLMCall(
                call_id=uuid.uuid4().hex,
                run_id=self._run_id,
                paper_id=paper_id,
                stage=stage,
                model=reply.model,
                request_hash=request_hash,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
                cache_read_input_tokens=usage.cache_read_input_tokens or 0,
                cost_usd=usd,
                latency_s=response.latency_s,
                attempts=response.attempts,
                status="refusal" if refused else "ok",
                stop_reason=reply.stop_reason,
                request_id=response.request_id,
                batch=response.batch,
                started_at=started_at,
            )
        )
        if refused:
            c.refusals += 1
            raise RefusalError(reply)
        if self._config.pipeline.use_cache:
            kind = "beta" if isinstance(reply, BetaMessage) else "message"
            self._store.cache_put(
                request_hash, {"kind": kind, "message": reply.model_dump(mode="json")}
            )
        return reply

    def _record_failure(
        self,
        model: str,
        request_hash: str,
        stage: str,
        paper_id: str | None,
        started_at: datetime,
        error: GatewayError,
    ) -> None:
        self._store.record_call(
            LLMCall(
                call_id=uuid.uuid4().hex,
                run_id=self._run_id,
                paper_id=paper_id,
                stage=stage,
                model=model,
                request_hash=request_hash,
                attempts=error.attempts,
                status="error",
                started_at=started_at,
            )
        )

    def _models_in_use(self) -> set[str]:
        agents = self._config.agents
        models = {role.model for role in agents.roles().values()}
        if self._config.pipeline.fallbacks:
            for model in list(models):
                models.update(FALLBACK_TARGETS.get(model, ()))
        return models


def _ewma(previous: float | None, value: float) -> float:
    if previous is None:
        return float(value)
    return (1 - _EWMA_WEIGHT) * previous + _EWMA_WEIGHT * value


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
