# Runtime: the store, the model gateway and the fake API

The runtime package is what every other package stands on. `precis.store`
keeps everything a run produces in one SQLite file. `precis.llm.Gateway` is
the only way any code talks to a model: it adds caching, a budget,
client-side rate limiting, retries, refusal fallbacks, cost accounting and
batching to every request. `precis.llm.fake.FakeAnthropic` stands in for the
Anthropic SDK client, so the whole pipeline (and this package's own
benchmarks) run without an API key, deterministically, in simulated time.

| Module | Responsibility |
|---|---|
| `store.py` | SQLite schema, migrations, typed reads and writes |
| `llm/gateway.py` | `Gateway`, `GatewayStats`: the request path below |
| `llm/limiter.py` | token buckets, the AIMD concurrency window, `Admission` |
| `llm/batch.py` | batch mode: queue, flush, poll, route results |
| `llm/pricing.py` | prices per model, `cost(usage, model, batch)` |
| `llm/models.py` | which models support server-side fallbacks, and to what |
| `llm/clock.py` | `Clock` protocol: real, scaled and manual time |
| `llm/client.py` | the SDK surface the gateway uses (a `Protocol`), `make_client` |
| `llm/errors.py` | `GatewayError`, `RefusalError`, `BudgetExceeded` |
| `llm/fake.py` | `FakeAnthropic`, `FakeReply`, latency, faults, server limits |

## The request path

```
create(params, stage, paper_id)
  │
  ├─ cache hit? ─────────────────────────────► LLMCall(status="cached", $0) ──► reply
  ├─ budget: spent + in flight + estimate > max? ──► BudgetExceeded
  │
  ├─ online ─┬─ wait for a concurrency slot (AIMD window)
  │          ├─ wait for request / input / output token buckets
  │          ├─ send via with_raw_response (beta endpoint + fallbacks if supported)
  │          ├─ 429 / 529 / 5xx / timeout ─► back off, release, try again (≤ max_attempts)
  │          └─ correct the reservation from usage, learn limits from headers
  │
  ├─ batch ──── queue ─► Message Batches API ─► poll ─► route result by custom_id
  │
  └─ LLMCall(cost from the serving model) ─► refusal? RefusalError : cache and return
```

`params` are exactly the keyword arguments of `client.messages.create`, so
agent code builds requests the way the SDK documents them and the gateway
never has to understand a prompt. The request hash is
`sha256(canonical_json(params))`; a stored response for the same hash is
returned without a request and recorded as a `cached` call that costs
nothing and counts no tokens (no tokens were processed). Refusals and
errors are never cached.

## Store

One SQLite file in WAL mode, opened in autocommit mode: every write is its
own short transaction, so a crash loses at most the requests that were in
flight, and readers (a `precis status` in another terminal) never block the
writer. Writes run inline on the event loop: recording an LLM call takes
about 45 µs on a laptop SSD (2,000 writes to a WAL file), less than the
overhead of handing it to a thread, and `usage()` over those 2,000 calls
takes under a millisecond.

Each record is stored as the JSON of its pydantic model and validated back
into that model on read, so a schema drift fails loudly at the boundary
instead of leaking half-parsed data into the pipeline. Next to the JSON sit
only the columns that queries filter, sort or aggregate on, each with an
index for the query that uses it: papers by `(published DESC, id)`, a
`paper_categories` table so `list_papers(category=...)` matches any listed
category through an index, stages by `(stage, status, paper_id)` for the
scheduler's "what is pending" query, and LLM calls by `(run_id, stage,
model)` so `usage()` is a `GROUP BY` in SQLite rather than a loop in Python.

The schema carries a version number. `MIGRATIONS[i]` upgrades version `i` to
`i + 1` inside one transaction; opening an older store runs the missing
migrations, and opening a store written by a newer version raises
`SchemaError` instead of guessing. The test suite exercises the path by
appending a migration and reopening an existing file.

## Admission control

The Messages API limits each model by requests, input tokens and output
tokens per minute, enforced with token buckets that refill continuously.
Rather than discover those limits through 429s, the gateway mirrors them.

**Token buckets.** `Admission` keeps one bucket per limit. A bucket holds
`burst_s` seconds of its rate (10 s by default, so a 600 RPM bucket holds
100 requests): the API documents that it may enforce a per-minute limit
over shorter intervals, and a full minute of burst would invite exactly the
429 the bucket exists to prevent. A request reserves its *estimated* input
and output tokens; when the response arrives the reservation is corrected to
what the API counted (uncached input plus cache writes; cache reads do not
count toward input-token limits). An underestimate puts the bucket in debt,
which the next requests wait off, so the long-run rate stays honest even
when single estimates are wrong. A request larger than the whole bucket is
let through when the bucket is full, rather than never.

Estimates come from `util.estimate_tokens` over the prompt, scaled by a
per-stage moving average of counted/estimated input (which falls well below
1 for stages that read most of their prompt from the cache), and from a
per-stage moving average of output tokens. One bug worth recording: the
first version measured that ratio against reservations that already
included the previous ratio, so with many requests in flight it compounded
until a reservation exceeded every bucket and admission stalled. The
closed-loop simulation below found it within two simulated minutes; the
ratio is now always taken against the raw prompt estimate, and the
convergence tests guard it.

**Learning the limits.** Each limit in `[limits]` (`rpm`, `itpm`, `otpm`)
can be set, or left `None` to be learned. Every response carries the
`anthropic-ratelimit-{requests,input-tokens,output-tokens}-{limit,remaining,reset}`
headers (names checked against the live rate-limits page on 2026-11-11);
the gateway reads them through `with_raw_response`, adopts the server's
limit (or keeps a configured one if it is lower), and lowers a bucket to the
server's `remaining` when the server knows of usage the client does not,
such as another process sharing the key. Until the first response a bucket
is open. A 429's `retry-after` pauses all admissions for that long, not just
the request that received it: the server's bucket is empty for everyone.

**AIMD concurrency.** The buckets bound the rate; a separate window bounds
how many requests are in flight. It starts at `initial_concurrency` (8),
grows by one after each window of `floor(limit)` successful responses, and
halves on a 429 or 529, at most once per congestion event (all requests that
were in flight when the first overload arrived belong to the same event, as
in TCP). Other failures (5xx, timeouts) say nothing about capacity and leave
the window alone. Why not a fixed concurrency? The right number depends on
latency, model and account, and changes during a run; a fixed setting is
either too low (idle capacity) or too high (a 529 storm). The trade-off is
throughput: halving means the window spends its time between half and all of
the capacity.

![AIMD limit and requests in flight over ten simulated minutes against a server that answers 529 above 24 concurrent requests](img/runtime-aimd.png)

Measured over ten simulated minutes against a fake that answers 529 to any
request arriving while more than 24 are in flight (64 closed-loop clients,
~4.5 s latency):

| Concurrency control | Successful requests/min | Responses that were 529 | Mean in flight |
|---|---|---|---|
| AIMD (8 → 64 ceiling) | 241 | 0.6 % | 18.1 |
| Fixed at 64 | 307 | 53.5 % | 24.1 |

A fixed window gets 27 % more through because the fake's overload answer is
nearly free (0.1 s) and retries land in the gaps; it does so by sending about
one failed request per successful one. Against the real API, where an
overloaded service is slow and every client's retries add to the load, the
AIMD sawtooth (one short 529 burst about once a minute here) is the
behaviour to want. The
halving factor is the knob if more throughput is wanted.

## Retries

The SDK client is built with `max_retries=0` (`make_client`) and the gateway
retries itself, because SDK retries happen behind the gateway's back: an
SDK-level retry of a 429 would not shrink the concurrency window, would not
pause other requests for `retry-after`, would not appear in `GatewayStats`
or in `LLMCall.attempts`, and its wait would not run on the injectable clock
the simulations depend on.

| Outcome | Retried | Window | Wait |
|---|---|---|---|
| 429 | yes | halve | `retry-after` (+ up to 1 s jitter), all admissions paused |
| 429 spend cap (`EnforcedSpendLimitReachedError`) | no | - | retrying cannot succeed until the cap resets |
| 529 | yes | halve | full-jitter exponential backoff |
| other 5xx, timeouts, connection errors | yes | unchanged | full-jitter exponential backoff |
| other 4xx | no | unchanged | - |

Full jitter waits a uniformly random time in `[0, min(60 s, 1 s · 2^(n-1))]`
before attempt `n + 1`, which spreads a burst of simultaneous failures
instead of retrying it in lockstep. The jitter comes from a `Random` seeded
by the run id, so a simulated run is reproducible. After `max_attempts` (8)
the request fails with `GatewayError` and is recorded as an `error` call.

## Refusals, fallbacks and the budget

With `pipeline.fallbacks` on, requests to models that support server-side
fallbacks (Claude Fable 5.2 and 5.1, Claude Opus 5.5 and 5, Claude Sonnet 5.5)
go through `client.beta.messages` with the `server-side-fallback-2026-07-01`
beta and `fallbacks="default"`, which lets the API pick the fallback model
by refusal category. Claude Haiku 5.5 has no server-side fallback, and the
Batches API rejects the parameter, so batch requests never carry it. A reply
whose final `stop_reason` is `refusal` raises `RefusalError` with
`stop_details.category`; it is recorded (refusals can be billed) but not
cached. Every model a fallback may route to is checked for a price when the
gateway starts.

`budget.max_cost_usd` is enforced before a request is sent: the gateway
refuses with `BudgetExceeded` when the money already spent, plus the
estimated cost of requests in flight, plus this request's estimate would
cross the cap. Counting in-flight estimates keeps 64 concurrent requests
from all passing the check at once and overshooting together.

## Pricing and accounting

`llm/pricing.py` holds per-million-token prices for input, output, 5-minute
and 1-hour cache writes and cache reads for every current model, from the
Anthropic pricing page (retrieved 2026-11-11), including two details that
are easy to get wrong: cache reads are 0.05x the input price on Claude
Opus 5.5 and Sonnet 5.5 (not the usual 0.1x), and Claude Haiku 5.5 prices a
whole request at a higher tier once its prompt (uncached, written and read
tokens together) exceeds 100,000 tokens. The Batch API halves every price.
A response is priced at the model that served it (`reply.model`), which
after a fallback is not the model requested. A model with no price is a
configuration error when the gateway starts, never a free call.

Every response becomes one `LLMCall` row: tokens by kind, cost, latency of
the last attempt (or, in batch mode, from queueing to result), the number of
attempts, the API request id, and whether it was a batch request. A known
gap: with a server-side fallback, the declined attempt's tokens are listed
in `usage.iterations`, but only the serving attempt (top-level `usage`) is
priced, because whether a declined attempt is billed depends on when and why
it declined.

## Batch mode

`Gateway(mode="batch")` serves the same `create` calls through the Message
Batches API at half price. Requests queue until `batch_max_requests` (500)
are waiting or `batch_max_wait_s` (30 s) after the first one; the batch is
then polled with doubling intervals from 10 s up to 120 s, its results are
streamed, and each result resolves the caller waiting on its `custom_id`.
An agent awaiting `create` cannot tell the difference, so the multi-turn
agents run unchanged: each turn is one request in some batch. The cost is
latency, minutes per turn instead of seconds, which is acceptable for an
overnight corpus run and not for an interactive one.

Failed items are retried **in a later batch**, not online: the caller has
already accepted batch latency, and an online retry would forfeit the
discount. `errored` items with a server-side error, `expired` and `canceled`
items, and items missing from the results are requeued together once a
batch's results have been read (so they share one batch instead of
trickling into several); `errored` items with `invalid_request_error` would
fail again and fail the caller at once. `aclose()` sends whatever is queued
and waits for every batch to finish.

Measured with 200 concurrent callers, `batch_max_requests=100`, and a fake
that errors 8 % and expires 4 % of batch items: the callers were served by
four batches of 100, 100, 21 and 2 requests; 21 requests needed a second
attempt and 2 a third; none failed; the cost was exactly half the cost of
the same 200 requests online.

## The fake API

`FakeAnthropic` implements `messages.create`, `messages.with_raw_response.create`,
the same under `beta.messages`, and `messages.batches.{create,retrieve,results}`,
and type-checks as the same `AsyncAnthropicLike` protocol as the real client.
It returns the SDK's own `Message`, `BetaMessage` and `MessageBatch` objects
and raises the SDK's own exception classes (`RateLimitError`,
`OverloadedError`, `APITimeoutError`, ...) around real `httpx2` responses
with real headers, so code under test takes the paths it takes in
production.

Replies come from *responders*, functions from request params to a
`FakeReply` (or `None` to pass); the first reply wins. A `FakeReply` is
either a complete SDK `Message` or content blocks plus a stop reason, with
helpers for text, tool use and refusals. Around the responders the fake
models what the runtime reacts to:

- **Usage.** Input from `estimate_tokens`; prompt caching as a prefix match
  over tools, system and message blocks, with explicit and automatic
  breakpoints, 5-minute and 1-hour TTLs, read refresh and the 512-token
  minimum; output from the reply's content, capped by `max_tokens`.
- **Server limits.** `ServerLimits(rpm, itpm, otpm)` buckets that refill
  continuously, answer 429 with `retry-after`, and send the rate-limit
  headers on every response; `set_limits` changes them mid-run.
- **Faults.** Random or scripted (`fail_next`) 429, 529, 500, timeouts and
  400s; 529 above a concurrency threshold; errored and expired batch items.
- **Validation.** The documented 400s for request shapes current models
  reject (`temperature`/`top_p`/`top_k`, thinking budgets, forced
  `tool_choice` on Claude Opus 5.5), 404 for unknown models, and a
  `LookupError` when no responder matches, so a test fails loudly.
- **Fallbacks.** On the beta path with `fallbacks`, a refusal is replayed on
  the fallback model and reported the way the API does (a `fallback` block,
  `fallback_message` in `usage.iterations`, the served model in `model`).

All randomness comes from one `Random(seed)`. Time comes from a `Clock`:
`ScaledClock(100)` runs fifteen simulated minutes in about nine seconds of
wall time with every component keeping its real-world parameters, and
`ManualClock` lets a unit test assert that a retry waited exactly
`retry-after`.

## Measurements

Closed-loop simulation (`uv run python -m tests.runtime.simulate`): 64
clients send `claude-opus-5-5` requests back to back (~1,000 input and 300
output tokens each, ~4.5 s latency with 30 % log-normal jitter) against a
fake with 600 RPM, 1.2M ITPM and 240K OTPM. Nothing is configured; every
limit is learned from headers. At 450 s the fake halves its request limit.

| Measure | Result |
|---|---|
| 429 responses | 0 of 6,144 |
| Requests/min, 240-450 s (window settled) | 605 of 600 |
| Requests/min, 60-450 s (includes the ramp) | 558 |
| Input / output tokens/min, 240-450 s | 606K / 181K |
| Time for the window to grow from 8 to 45 (enough for 600 RPM) | 176 s |
| New 300 RPM limit adopted after the change | 1.9 s (the next response) |
| 30-s windows after the change | mean 300 RPM, max 316 RPM, 0 429s |
| Failed requests | 0 |

The client runs at the limit without a single 429 because it reads the
limit from the first response and spends its burst conservatively; the
0.8 % over in steady state is the 10-second burst spent on top of the refill,
which the server's full-minute bucket absorbs. The slow part is the
additive increase from the initial window: raising `initial_concurrency`
shortens the ramp at the risk of an early overload burst.

## Limitations

- Limits are tracked per gateway, not per model; a run that mixes models
  with separate rate-limit pools is limited by the tightest one.
- The AIMD window counts requests waiting for bucket capacity as in flight,
  so when the buckets are the bottleneck the window can grow to its ceiling
  without that load reaching the server; it is bounded by `max_concurrency`.
- Cached calls record zero tokens, so token totals measure API usage, not
  the size of the work; use `cached_calls` to see how much was replayed.
- The fake's cache model ignores the 20-block lookback window and the rule
  that an entry becomes readable only once its writer starts streaming.
