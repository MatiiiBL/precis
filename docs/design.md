# Précis design

Précis turns a stream of research papers into verified, structured summaries
and a map of the field. A run fetches a few thousand papers from arXiv, has an
LLM agent read each one and write a structured summary, has a second agent
check every claim in that summary against the paper, rewrites the summaries
that fail, and then clusters the corpus into topics and writes a report and a
static website. Every request goes through one gateway that does admission
control, retries, cost accounting and response caching, so a 2,000-paper run
can be stopped, resumed and re-run without paying twice.

This document is the contract between the packages. Interfaces, record
shapes, CLI flags and file layouts described here change only together with
this file.

## 1. Pipeline

```
                 per paper (thousands, in parallel)                     corpus (once)
 ┌───────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────┐  ┌────────────┐  ┌──────┐
 │ fetch │──▶│ read (agent) │──▶│verify (agent)│──▶│revise (agent)│──▶│ topics │─▶│ synthesize │─▶│ site │
 └───────┘   └──────────────┘   └──────────────┘   └──────────────┘   └────────┘  └────────────┘  └──────┘
  arXiv API    navigates the      checks each        rewrites only      TF-IDF +     one agent      static
  HTML / PDF   paper with tools,  claim against the  summaries with     k-means,     call per       HTML +
  → Document   → PaperSummary     text → Verification failed claims     LLM labels   topic + report search
```

Per-paper stages (`models.Stage`): `fetch`, `read`, `verify`, `revise`.
Each stage's result is stored as soon as it exists, keyed by
`(paper_id, stage)`, together with a `config_hash` of everything that
determines it (model, effort, prompt version, strategy, options). A stage is
skipped when a `done` record with the same hash exists, so re-running a
command only does the missing or changed work.

`revise` runs only when the verification has an `unsupported` or
`contradicted` verdict, at most once per paper. Its output replaces the
summary (`revision = 1`), and the new summary is verified again so the report
can show faithfulness before and after revision.

Stage outputs (`StageRecord.output`):

| Stage | Output | Notes |
|---|---|---|
| `fetch` | `{version, source_format, n_tokens, n_sections, n_captions, fallback_reason}` | the `Document` itself is stored with `put_document` |
| `read` | `PaperSummary` (revision 0) | |
| `verify` | `Verification` of revision 0 | never overwritten |
| `revise` | `RevisionResult` (summary revision 1 and its verification) | `skipped` when the verification passed or revision is off |

A paper's final summary is the `revise` summary when that stage is `done`,
and the `read` summary otherwise; its final verification follows the same
rule.

## 2. Records

`src/precis/models.py` defines every record that crosses a package boundary
(`PaperMeta`, `Document`, `PaperSummary`, `Verification`, `RevisionResult`, `Topic`,
`TopicSynthesis`, `CorpusReport`, `StageRecord`, `LLMCall`, `UsageRow`).
Read it; the field descriptions are part of this contract.

## 3. Packages and ownership

| Path | Package | Purpose |
|---|---|---|
| `src/precis/models.py`, `util.py`, `config.py`, `cli/__init__.py` | core | records, helpers, config, CLI root |
| `src/precis/llm/`, `src/precis/store.py` | runtime | gateway, rate limiting, retries, pricing, cache, batching, fake API; SQLite store |
| `src/precis/sources/`, `src/precis/text/`, `cli/fetch.py` | ingest | arXiv search and download, HTML/PDF parsing, sectioning, BM25 search |
| `src/precis/agents/`, `src/precis/pipeline/`, `src/precis/offline.py`, `cli/run.py` | agents | reader, critic, reviser, summarization strategies, scheduler, progress |
| `src/precis/corpus/`, `src/precis/eval/`, `cli/corpus.py`, `cli/evaluate.py` | corpus | clustering, topic synthesis, report, site; evaluation and benchmarks |

Each package has its own tests under `tests/<package>/` and its own document
under `docs/` (`runtime.md`, `ingest.md`, `agents.md`, `evaluation.md`).
Each section of `config.py` belongs to the package that reads it.

## 4. Store (`precis.store`)

One SQLite file (`Config.store_path`, default `data/precis.db`) in WAL mode
holds everything a run produces. All pipeline code runs on one asyncio event
loop and calls the store from that thread; writes are small and committed
immediately, so a crash loses at most the requests that were in flight.

```python
class Store:
    def __init__(self, path: Path | str) -> None: ...      # ":memory:" for tests
    def close(self) -> None: ...

    # papers and text
    def upsert_paper(self, meta: PaperMeta) -> None: ...
    def get_paper(self, paper_id: str) -> PaperMeta | None: ...
    def list_papers(self, *, category: str | None = None,
                    limit: int | None = None) -> list[PaperMeta]: ...   # newest first, then id
    def count_papers(self) -> int: ...
    def put_document(self, doc: Document) -> None: ...
    def get_document(self, paper_id: str) -> Document | None: ...

    # per-paper stages (upsert by (paper_id, stage))
    def set_stage(self, record: StageRecord) -> None: ...
    def get_stage(self, paper_id: str, stage: Stage) -> StageRecord | None: ...
    def stage_counts(self) -> dict[Stage, dict[StageStatus, int]]: ...
    def papers_with_status(self, stage: Stage, status: StageStatus) -> list[str]: ...
    def reset_running(self, stages: Iterable[Stage] | None = None) -> int: ...
        # RUNNING -> PENDING after a crash, only for the stages the caller owns
        # (None = all); precis run passes read/verify/revise, precis fetch FETCH

    # LLM calls and the response cache
    def record_call(self, call: LLMCall) -> None: ...
    def calls(self, *, run_id: str | None = None,
              paper_id: str | None = None) -> list[LLMCall]: ...
    def usage(self, *, run_id: str | None = None) -> list[UsageRow]: ...
    def cache_get(self, request_hash: str) -> dict[str, Any] | None: ...
    def cache_put(self, request_hash: str, response: dict[str, Any]) -> None: ...
    def cache_delete(self, request_hash: str) -> None: ...

    # message batches, recorded from creation until their results are read
    def record_batch(self, batch_id: str, items: Sequence[BatchItem]) -> None: ...
    def open_batches(self) -> list[tuple[str, list[BatchItem]]]: ...
    def close_batch(self, batch_id: str) -> None: ...

    # runs and corpus artifacts (topics, syntheses, reports, eval results)
    def start_run(self, run_id: str, config: dict[str, Any]) -> None: ...
    def finish_run(self, run_id: str, stats: dict[str, Any]) -> None: ...
    def runs(self) -> list[dict[str, Any]]: ...
    def put_artifact(self, key: str, value: Any) -> None: ...   # BaseModel or JSON value
    def get_artifact(self, key: str) -> Any | None: ...          # parsed JSON
```

The schema has a version number; opening a store with an older schema
migrates it.

## 5. LLM runtime (`precis.llm`)

All model traffic goes through `Gateway.create`. No other module calls the
Anthropic SDK directly.

```python
class Gateway:
    def __init__(self, client: AsyncAnthropicLike, store: Store, *, run_id: str,
                 config: Config, clock: Clock | None = None,
                 mode: Literal["online", "batch"] = "online") -> None: ...
    async def create(self, params: dict[str, Any], *, stage: str,
                     paper_id: str | None = None) -> Reply: ...  # Message | BetaMessage
    @property
    def spent_usd(self) -> float: ...       # this run
    @property
    def total_spent_usd(self) -> float: ... # everything in the store: what the budget caps
    def stats(self) -> GatewayStats: ...    # in flight, concurrency limit, calls,
                                            # retries, 429s, tokens, cost, ...
    async def aclose(self) -> None: ...     # stop; unfinished batches stay recorded
    @contextmanager
    def cache_scope(self, salt: str) -> Iterator[None]: ...  # see Cache below

def make_client(config: Config) -> AsyncAnthropicLike: ...   # real SDK client, max_retries=0
```

`params` is exactly the keyword arguments of `client.messages.create`
(`model`, `max_tokens`, `system`, `messages`, `tools`, `output_config`, ...).
The gateway adds the cross-cutting parts:

1. **Cache.** The request hash is `sha256(canonical_json(params))`. With
   `pipeline.use_cache`, a stored response for the same hash is returned
   without a request and recorded as a `cached` call costing 0. Prompts must
   therefore be deterministic: no timestamps, ids or unordered collections.
   Refusals and replies with `stop_reason == "max_tokens"` are never cached,
   and a stored row that no longer reads back is deleted and treated as a
   miss. Inside `with gateway.cache_scope(salt)`, a non-empty `salt` is mixed
   into the hash of every request (tasks started inside inherit it): the
   scheduler wraps the re-execution of a `failed` stage in
   `cache_scope(f"attempt-{n}")`, so the retry gets fresh responses while a
   crash during that attempt still resumes from its own cache. The empty
   salt is the plain hash above.
2. **Admission.** A request waits for (a) a concurrency slot under an AIMD
   limit (start at `limits.initial_concurrency`, +1 per window of successes up
   to `limits.max_concurrency`, halve on 429/529), and (b) token-bucket
   capacity for requests, input tokens and output tokens per minute. Buckets
   use the configured limits, or learn them from the
   `anthropic-ratelimit-*` response headers when a limit is `None`. Input is
   reserved from `util.estimate_tokens`, output from a running average per
   stage; both are corrected from `usage` when the response arrives.
3. **Retries.** Every request is streamed (`messages.stream` and
   `get_final_message()`), so long generations do not hit the HTTP timeout,
   which bounds each read rather than the whole reply. 429, 529, other 5xx
   and connection errors retry with exponential backoff and full jitter,
   honouring `retry-after`, up to `limits.max_attempts`; a timeout is
   retried once (the server may already have generated and billed the
   reply) and its request id is logged. Other 4xx errors fail immediately.
   The SDK's own retries are off (`max_retries=0`) so that every attempt is
   visible to admission control and accounting.
4. **Refusals.** With `pipeline.fallbacks`, requests to models that support
   server-side fallbacks carry them. A final `stop_reason == "refusal"`
   raises `RefusalError`; the pipeline marks the stage `refused` and moves on.
5. **Accounting.** Each request is recorded once as an `LLMCall`: the reply
   that came back (status `ok` or `refusal`, with the number of attempts it
   took), a `cached` hit, or an `error` row when it failed for good; the
   failed attempts before a success are counted in `attempts`, not recorded
   separately. Its cost comes from `precis.llm.pricing` (input, output,
   cache writes and cache reads priced separately; each entry of
   `usage.iterations`, such as an attempt declined before a fallback, at its
   own model; batch requests at the batch discount).
   **Budget.** `budget.max_cost_usd` caps the total spend recorded in the
   store, earlier runs included. Before each request the gateway reserves
   its uncached input at the request's cache-write rate and its output at
   the largest reply its stage has produced (its whole `max_tokens` until the
   stage has one), and raises `BudgetExceeded` when spent plus in-flight
   reservations would cross the cap (saying so when the cap cannot fit even
   one request); the scheduler then stops admitting papers. Replies longer
   than their reservation can overshoot by at most about
   `max_concurrency x max_tokens x output price`.
6. **Batching.** In `mode="batch"` the gateway queues requests and sends them
   through the Message Batches API (flush at N requests, at
   `limits.batch_max_bytes`, or after T seconds; poll; route each result back
   to its waiting caller by `custom_id`). Callers are unchanged: an agent
   awaiting `create` does not know whether it was served online or by a
   batch, so the same multi-turn agents run at the batch price. Results are
   never lost or bought twice: each batch is recorded in the store
   (`record_batch`) when created, every succeeded result is recorded and
   cached even if its caller has gone, polling and result errors retry the
   same batch, and a gateway collects batches earlier runs left open (a
   request whose result is in one waits for it rather than being resent).
   `aclose` does not wait for running batches; it leaves them recorded and
   logs their ids. A batch the API rejects with a 4xx is split in half.

`precis.llm.fake.FakeAnthropic` implements the parts of the SDK client the
gateway uses (`messages.stream`, `messages.create`,
`messages.with_raw_response.create`, the same under `beta.messages`, and
`messages.batches`), returning real SDK `Message` objects. Behaviour comes
from a list of responders (`Callable[[dict[str, Any]], FakeReply | None]`;
the first non-`None` reply wins), plus a latency model, fault injection (429
with `retry-after`, 529, 500, timeouts, errors inside a started stream,
Batches API outages and size limits) and optional simulated server rate
limits. Time comes from an
injectable `Clock`; a scaled clock runs a simulated hour in seconds, which is
what makes the throughput benchmarks fast and deterministic.

Model facts the runtime and the agents rely on (current Claude models;
details in the Anthropic API docs):

- The default model for every role is `claude-opus-5-5`; each role's model is
  configurable (`[agents.reader] model = ...`).
- Thinking is adaptive and on by default on current models; depth is set with
  `output_config={"effort": ...}`. Do not send `budget_tokens`,
  `temperature`, `top_p` or `top_k`.
- `tool_choice` of type `any` or `tool` is rejected by current models. Use
  `auto`, name the tool in the prompt, and give tools `strict: true`.
- In a tool loop, append `response.content` unchanged (thinking blocks
  included) and return every `tool_result` of a turn in one user message.
- Prompt caching: send top-level `cache_control={"type": "ephemeral"}` so
  each turn of a loop reads the previous turn's prefix from cache.

## 6. Ingest (`precis.sources`, `precis.text`)

- `sources.arxiv` searches the arXiv API by category and submission date
  range (Atom feed, paged); `sources.fulltext` downloads full text. HTML from
  `https://arxiv.org/html/<id>v<version>` is preferred because it keeps the
  section structure; the PDF is the fallback. All arXiv requests share one
  limiter (`fetch.min_interval_s`, 3 s by default) and send
  `fetch.user_agent`. Raw files are kept under `fetch.raw_dir`, so parsing
  can be repeated without downloading again.
- `text`: HTML (LaTeXML) and PDF to `Document`: section tree with stable ids,
  math as its LaTeX source, figure and table captions, no bibliography and no
  abstract. `text.bm25.PaperIndex(doc)` splits a document into passages of
  about 120 words and answers `search(query, k) -> list[Passage]`, where a
  passage has `id`, `section_id`, `text` and `score`.
- `sources.local`: a directory of PDFs, for papers that are not on arXiv.

## 7. Agents and pipeline (`precis.agents`, `precis.pipeline`)

```python
async def read_paper(gw: Gateway, meta: PaperMeta, doc: Document,
                     config: AgentsConfig) -> PaperSummary: ...
async def verify_summary(gw: Gateway, meta: PaperMeta, doc: Document,
                         summary: PaperSummary, config: AgentsConfig) -> Verification: ...
async def revise_summary(gw: Gateway, meta: PaperMeta, doc: Document, summary: PaperSummary,
                         verification: Verification, config: AgentsConfig) -> PaperSummary: ...
```

- **Reader** (`strategy = "agentic"`). The first message gives the title,
  authors, the abstract when `include_abstract` is on, and the outline
  (section ids, titles, token counts). Tools: `read_section(section_id,
  page)`, `search_paper(query)`, `read_captions()` and
  `submit_summary(<SummaryDraft>)`, which ends the loop after validation (a
  validation error goes back to the model as an error `tool_result`). The
  agent decides what to read. When `max_turns` or `token_budget` is nearly
  used up, a mid-conversation system message tells it to submit.
- **Other strategies**, for the ablation: `fulltext` (one request with the
  whole body, structured output), and `mapreduce` (notes per chunk in
  parallel, then one combining request). All three produce `PaperSummary`.
- **Critic.** Gets the summary and the same reading tools, and must submit a
  `ClaimCheck` for the tldr and for every finding, with a verbatim quote as
  evidence.
- **Reviser.** Gets the summary, the failed checks and the reading tools, and
  submits a corrected `SummaryDraft`.
- Prompts live in `src/precis/agents/prompts/*.md`; a prompt's version is the
  hash of its text and is part of the stage `config_hash`.
- **Scheduler.** `Pipeline(config, store, gateway).run(paper_ids, stages)`
  keeps at most `pipeline.max_papers_in_flight` papers active (the gateway
  regulates the request rate underneath), runs each paper's stages in order,
  stores each result immediately, and records failures per stage. On SIGINT
  it stops admitting papers, cancels in-flight work and leaves those stages
  `pending`; `Store.reset_running` repairs a store after a hard crash. A live
  progress view shows stage counts, requests in flight, the concurrency
  limit, tokens, spend and ETA.
- `precis.offline` builds a `FakeAnthropic` whose responders play every role
  deterministically from the document text, so every command runs with
  `--offline`, without an API key; CI uses it for the end-to-end tests.

## 8. Corpus, evaluation and benchmarks (`precis.corpus`, `precis.eval`)

- **Topics.** TF-IDF over each summary's tldr, approach, methods and
  keywords, reduced with truncated SVD and clustered with k-means (k chosen
  by silhouette score unless given). One LLM request per cluster names it
  and describes it from its most central papers.
- **Synthesis.** One request per topic over the compact summaries of its
  papers writes a `TopicSynthesis` that cites papers as `[arXiv:<id>]`; a
  citation of a paper outside the topic is removed and counted. A last
  request writes the report introduction. `precis report` writes Markdown.
- **Site.** `precis site` renders a static site (Jinja2): overview, one page
  per topic, one per paper (summary, verification verdicts, links), and
  client-side search over a JSON index, with no external scripts.
- **Evaluation.** `precis eval abstract`: summaries written without the
  abstract are compared with it (ROUGE, and key-point recall judged by a
  model). `precis eval faithfulness`: verdict rates before and after
  revision. `precis eval ablation`: the three strategies on the same papers
  (tokens, cost, latency, recall, faithfulness).
- **Benchmarks.** `precis bench throughput` runs the full per-paper pipeline
  against `FakeAnthropic` with simulated rate limits and latency across a
  sweep of concurrency limits; `precis bench faults` injects errors and a
  crash mid-run and checks that no paper is lost and no completed request is
  paid for twice.

## 9. Command line

```
precis fetch arxiv --category cs.AR [--category ...] --since 2025-07-01 [--until ...] [--limit N]
precis fetch ids 2406.01234 ...
precis fetch local <dir>
precis run [--stages read,verify,revise] [--strategy agentic|fulltext|mapreduce]
           [--limit N] [--ids ...] [--mode online|batch] [--max-cost USD] [--offline]
precis status
precis topics [--k N]
precis report [--out reports/<name>.md]
precis site [--out build/site]
precis export --out summaries.jsonl
precis eval abstract|faithfulness|ablation [--n N]
precis bench throughput|faults
```

Global options: `--config` (TOML, default `./precis.toml`), `--store`,
`--verbose`. Commands that make requests accept `--offline`.

## 10. Conventions

- Python 3.12+, managed with `uv`. `make check` runs `ruff check`,
  `ruff format --check`, `mypy` (strict, on `src/`) and `pytest`.
- Tests never touch the network or the real API by default. Tests that need
  arXiv are marked `network`; tests that call the Anthropic API are marked
  `live` and skip without credentials.
- Runtime data lives under `data/` (ignored by git). Outputs meant to be
  published (reports, benchmark results, figures) live under `results/` and
  `docs/img/`.
