# Agents and the pipeline scheduler

This package turns a parsed paper into a verified summary. A **reader**
agent navigates the paper with tools and writes a `PaperSummary`; a
**critic** agent checks the tldr and every finding against the text, with a
verbatim quote for each; a **reviser** rewrites the summaries that fail, and
the critic checks the rewrite. The **scheduler** runs those stages over
thousands of papers, resumably, and shows a live view of the run. All model
traffic goes through `precis.llm.ModelGateway`, so the same code runs online,
through the Message Batches API, or against the offline fake.

| Module | Responsibility |
|---|---|
| `agents/paper.py` | what the agents see of a paper: header, outline, pages; quote and number matching |
| `agents/tools.py` | `read_section`, `search_paper`, `read_captions` |
| `agents/loop.py` | the tool loop: budgets, nudge, submission retries, stop reasons |
| `agents/messages.py` | request shape, replies, `max_tokens` retry, refusals |
| `agents/submissions.py`, `agents/schema.py` | submit tools, strict schemas, validation |
| `agents/reader.py`, `critic.py`, `reviser.py` | the three roles |
| `agents/strategies.py` | `fulltext` and `mapreduce` baselines, `summarize` dispatch |
| `agents/prompts/*.md` | system prompts; their hash is their version |
| `agents/offline.py`, `precis/offline.py` | deterministic responders and the `--offline` client |
| `pipeline/scheduler.py` | `Pipeline`: stage order, skipping, failures, budget, SIGINT |
| `pipeline/hashing.py` | the `config_hash` of each stage |
| `pipeline/stats.py`, `pipeline/progress.py` | run statistics and the live view |
| `cli/run.py` | `precis run`, `precis status`, `precis export` |

Public entry points are `summarize` (the configured strategy), `read_paper`,
`verify_summary` and `revise_summary`, as specified in the design. Each has
a `run_*` twin that also returns an `AgentTrace` (requests, tokens, tool
calls, rejected submissions, share of the paper read), which the scheduler
aggregates into the run record.

## 1. Why the reader navigates

A paper's value is unevenly spread: the contribution is stated in the
introduction, the mechanism in a method section, the evidence in a results
section and a few tables; related work, background, proofs and appendices
rarely change a summary. The reader therefore starts from the outline, not
from the text. Its first message holds the title, authors, categories, the
abstract (unless `include_abstract` is off) and one line per section with its
id, title and size:

```
Body: 8,004 tokens in 7 sections. read_section returns pages of about 2,500 tokens.
s1  Introduction  (986 tokens, 1 page)
s2  Background and Related Work  (951 tokens, 1 page)
s3  The Cache85 Design  (1,249 tokens, 1 page)
  s3.1  Overview  (1,287 tokens, 1 page)
s4  Evaluation  (2,440 tokens, 1 page)
s5  Discussion and Limitations  (1,055 tokens, 1 page)
s6  Conclusion  (36 tokens, 1 page)
Captions: 1 figures, 1 tables (read_captions).
```

From there it reads what it needs. Two consequences follow. The cost of a
summary scales with what is read rather than with paper length, which keeps
the long tail cheap; and the model's attention is spent on the sections that
carry results, not diluted over 20,000 tokens of background. The price is
latency (several sequential turns instead of one request) and the risk that
the reader misses something it did not open, which is what the critic and
the `search_paper` tool are for.

Section 7 has the measured numbers. In short, on the synthetic corpus the
offline reader opened 53% of each body, and its tokens grew by 0.31 per
extra body token against 1.0 for full text: it spends more than full text on
papers under about 7,000 tokens (where re-sending the conversation costs
more than the sections it skips) and much less on long ones. The ablation
(`precis eval ablation`) compares quality on real papers.

## 2. Tools

| Tool | Input | Returns |
|---|---|---|
| `read_section` | `section_id`, `page` | one page of about 2,500 tokens: `[s3] Design (page 1 of 2)`, the text, and what comes next |
| `search_paper` | `query` | the 5 best BM25 passages (`precis.text.bm25.PaperIndex`) with section ids and titles |
| `read_captions` | — | every figure and table caption with its id |
| `submit_summary` / `submit_verification` | the record | ends the run if the submission validates |

Design choices, each aimed at keeping the model on track without a human:

- **Results are self-locating.** Every page says "page i of n" and how to
  continue; the last page names the next section in the outline. A heading
  with no text of its own lists its subsections instead of returning nothing.
- **Mistakes are tool results, not exceptions.** An unknown id comes back
  as an error `tool_result` with close matches ("Did you mean s4.1?") and the
  full list of valid ids; a page out of range says how many pages there are;
  an empty query asks for keywords. The model corrects itself on the next
  turn instead of the run failing.
- **Tables are reachable.** `read_captions` shortens long captions to keep
  its result near one page, and `read_section` accepts caption ids
  (`"tab2"`) to return one caption in full, cell text included, since tables
  hold most headline numbers.
- **Re-reading is free.** A page already returned in this conversation is
  answered with a one-line pointer to it, not with the text again.
- **Strict schemas.** Every tool has `strict: true`. Strict mode rejects
  numeric, length and array-size constraints, `$ref`s are best avoided, and
  Pydantic's field descriptions are written for developers, so
  `agents/schema.py` derives each schema from its record, inlines references,
  drops unsupported keywords, makes every property required (optional ones
  stay nullable) and substitutes descriptions written for the model. A test
  walks every schema and fails on any unsupported keyword.

The tool set is identical for every paper and never changes during a
conversation, so tools and system prompt form a prefix that is cached once
and reused by every paper of the role.

## 3. The tool loop

The loop (`agents/loop.py`) is written by hand rather than with the SDK's
tool runner because requests must go through the gateway (admission,
accounting, batching) and because it enforces things the runner does not.

- **Requests** carry the role's model and `output_config.effort`, adaptive
  thinking, the reading tools plus the role's submit tool, and no
  `temperature` or `tool_choice`: current models reject sampling parameters
  and forced tool use, so the prompt names the submit tool and `strict`
  keeps its input well-formed.
- **The conversation is append-only.** The assistant turn goes back exactly
  as it came, thinking blocks and signatures included; it is serialized to
  JSON (with `None` fields dropped) rather than passed as SDK objects, so that
  a reply served live and the same reply restored from the response cache
  produce identical next requests, and therefore identical request hashes.
  All `tool_result`s of a turn go back in one user message, in call order.
- **Caching.** The system prompt carries an explicit breakpoint (shared by
  every paper of the role) and the request sets top-level automatic caching,
  which moves a breakpoint to the end of the conversation each turn, so each
  turn reads the previous one from the cache. Offline, cache reads made up
  50% of the reader's tokens and 37% of the critic's.
- **Budgets.** When two turns remain of `max_turns`, or 80% of
  `token_budget` is spent (input, cache and output tokens summed over the
  run), one mid-conversation `system` message tells the model to stop
  reading and submit, naming the submit tool. A system message rather than an
  edit to the system prompt keeps the cached prefix intact and speaks with
  operator authority. Once the budget is spent, reading tools return an error
  saying so; two grace turns allow a submission or a correction, then the
  run fails.
- **Nudge.** If the model ends its turn without calling the submit tool,
  one user message tells it that nothing is recorded until it does. A second
  such turn fails the run.
- **Submission errors** return as an error `tool_result` that lists every
  problem at once (section 4), so one correction turn usually suffices.

Stop reasons are handled deliberately:

| `stop_reason` | Handling |
|---|---|
| `tool_use` | run the tools, or accept the submission |
| `end_turn`, `stop_sequence` | nudge once, then fail |
| `max_tokens` | discard the truncated reply (a cut-off tool call or JSON is unusable) and resend once with twice the limit; thinking counts toward `max_tokens`, so a long deliberation is the usual cause |
| `refusal` | raise `AgentRefused`; the stage is recorded `refused` (the gateway raises `RefusalError` itself when server-side fallbacks also declined) |
| `pause_turn` | append and resend (only server tools pause; none are used) |
| anything else | fail with the stop reason in the error |

## 4. Prompts and validation

Each role's system prompt is a Markdown file in `agents/prompts/`. The four
prompts that write summaries include `summary_style.md`, which asks for
specific, quantitative, plain technical English: name the baseline, metric,
workload and setting of every number; copy numbers exactly and never
estimate, round, convert or compute one; keep the paper's scope; no
promotional adjectives; expand acronyms. The critic's prompt defines the four
verdicts operationally and is strictest about numbers and scope: a number
the paper does not state makes a claim at most `partial`, a different value
for the same quantity makes it `contradicted`, and a broader scope than the
paper's ("on all benchmarks" for two) is `partial`.

Prompts contain nothing paper-specific, so they are a cache prefix shared by
every request of a role. A prompt's version is the first 12 hex digits of
the SHA-256 of its final text and is part of the stage `config_hash`
(section 6), so editing a prompt re-runs exactly the stages that use it.

What a schema cannot express is validated in code before a submission is
accepted (`agents/submissions.py`):

- tldr at most 40 words; 2-6 findings; 3-8 keywords;
- every finding cites at least one id, and every cited id exists (a section
  or a caption);
- **every number** in the tldr, problem, approach, contributions, limitations
  and findings appears somewhere in the paper (title, abstract, body or
  captions), after canonicalizing (`1,024` = `1024`, `1.50` = `1.5`); integers
  up to 12 are exempt because papers spell them out. This is the cheapest
  defence against invented and rounded numbers, the critic's most common
  finding, and it costs one turn only when it fires.

Harmless differences are fixed silently instead of costing a turn: list
items are stripped and de-duplicated and keywords lowercased.

## 5. Verification and revision

The critic gets the outline and the claims, labelled `tldr`,
`finding:0`, ... with the sections each finding cites, but not the abstract,
which is not evidence. It must submit exactly one `ClaimCheck` per claim; a
submission with a missing, duplicate or unknown target, an unknown
`section_id`, or a non-`unsupported` verdict without a quote is rejected
with the full list of problems.

Models can produce fluent "quotes" that do not occur in the source. After
the critic submits, each quote is looked up in the paper with whitespace,
case, typographic quotes and dashes, and Unicode compatibility forms
normalized, ignoring edge punctuation, and treating "..." as a gap between
fragments that must each occur. A `supported` verdict whose quote is not
found is downgraded to `partial`, and its note is prefixed with
`[quote not found in the paper]`, so the stored record shows why and
`quotes_not_found()` counts it; the run statistics report the total.

The reviser runs when a check is `unsupported` or `contradicted` (and
`agents.revise` is on). It gets the summary as JSON and every check that was
not `supported`, with the critic's notes and quotes presented as leads to
confirm, and the same reading tools: it corrects claims from the text, not
from the critic's note, because the critic can be wrong too. Its summary
passes the same validation, and the critic then verifies it inside the same
stage; the stage output is a `RevisionResult` (revision 1 and its
verification), so faithfulness before and after revision can be compared.

## 6. Scheduling

`Pipeline(config, store, gateway).run(paper_ids, stages)`:

- **Bounded work.** `max_papers_in_flight` workers each take the next paper
  from a queue; the gateway regulates the request rate underneath, so this
  bound only limits how much work and memory are open at once.
- **Stage order and inputs.** A paper runs `read`, `verify`, `revise` in
  order, up to the last stage requested. A stage that is not requested only
  supplies its stored result (`--stages verify` verifies whatever summaries
  exist); a paper whose input is missing is counted as blocked.
- **Skipping.** A requested stage is skipped when a `done` (or, for revise,
  `skipped`) record with the same `config_hash` exists. The hash covers the
  role settings (model, effort, limits), the prompt versions, a version of
  the tool definitions and schemas, the options that shape requests (page
  size, search results, strategy, chunk size, whether the abstract is shown),
  and the stage's inputs: the document and metadata shown, the summary, the
  verification. A changed summary therefore re-runs its verification even if
  no setting changed. When a stage produces a different result than before,
  the stored results of later stages are marked `pending` so they are never
  read as current.
- **Failures.** Each stage is recorded `running` before it starts and
  `done`, `failed` (with the exception's type and message), or `refused`
  after; a failure stops that paper's later stages and the run goes on.
  `attempts` counts how often a stage was tried. `Store.reset_running` turns
  stages left `running` by a crash back to `pending` at the start of a run.
- **Budget.** `BudgetExceeded` from the gateway leaves the stage `pending`
  with the reason and stops admitting papers; papers in flight finish if their
  requests still fit, and the next run (with a larger budget) continues.
- **SIGINT.** The first Ctrl-C stops admission and cancels in-flight work;
  interrupted stages go back to `pending` with the error `interrupted`, the
  run record is finished, and the exit code is 130.

Each run is recorded with its configuration and statistics: papers finished
and blocked, per-stage outcomes (done, cached, skipped, failed, refused,
interrupted), per-role agent statistics (mean requests and tokens, median
tokens, share of the paper read, tool calls, rejected submissions, nudges,
budget notices, `max_tokens` retries, quotes not found), faithfulness before
and after revision, and the gateway's counters. `precis run` shows a live
view from the same counters and `Gateway.stats()`, refreshed four times a
second without touching the store: per-stage counts, requests in flight and
queued, the concurrency limit, calls, retries, rate limits, tokens, spend
against the budget, and an ETA from the rate at which papers finish.

`precis status` prints stage counts, recent runs and usage by stage and
model; `precis export --out summaries.jsonl` writes each paper's final
summary and verification, read through `precis.corpus.load`.

## 7. Measurements (offline)

**These numbers describe the machinery, not a model.** The offline
responders (section 8) play each role with fixed heuristics: the reader
always opens the first page of the introduction, the method and the results.
What they measure is how requests, tokens and caching behave for a given
reading pattern; how much a real model chooses to read is bounded by
`max_turns` (10) and `token_budget` (80,000) and is reported per run in the
same statistics.

Setup: 200 papers from `precis.eval.synthetic` (median body 8,540 tokens),
all three stages, `FakeAnthropic` usage and prompt-cache model at
`claude-opus-5-5` prices, online mode, the BM25 index from `precis.text`.

| Strategy, read stage | requests / paper | tokens / paper, median | p90 | share of body read | read cost (200 papers) |
|---|---|---|---|---|---|
| agentic | 2.2 (2 + 1 when a submission is rejected) | 10,113 | 18,420 | 53% | $7.35 |
| fulltext | 1 | 9,748 | 16,362 | 100% | $8.90 |
| mapreduce | 2.7 | 10,963 | 17,928 | 100% | $10.39 |

Tokens add input, cache writes, cache reads and output over every request of
the stage. The agentic reader's tokens are dominated by re-sending the
conversation, but half of them are cache reads at a twentieth of the input
price, which is why it is the cheapest of the three at a similar token
count. Its p90 includes the 41 papers (20%) whose first submission was
rejected and corrected.

Paper length matters more than the medians suggest. Regressing per-paper
read tokens on body tokens (agentic: the 159 papers read in two turns):

| Strategy | extra tokens per body token | median, bodies < 7,000 tokens | median, bodies > 12,000 tokens |
|---|---|---|---|
| agentic | 0.31 | 8,153 | 11,699 |
| fulltext | 1.00 | 6,661 | 16,283 |
| mapreduce | 1.07 | 7,509 | 17,889 |

Navigation costs a fixed overhead (outline, tool definitions, a second turn
that re-sends the first) and then scales with what is read, not with what
exists; full text scales with the paper. The synthetic lengths follow arXiv
computer-architecture papers (median about 8,500 tokens, 95th percentile
about 17,500), so the savings come from the long tail.

The rest of the agentic run:

| Stage | requests / paper | tokens / paper, median | share of body read | cost (200 papers) |
|---|---|---|---|---|
| verify (critic) | 2 | 8,420 | 38% | $6.47 |
| revise (reviser + critic, 37 papers) | 4 | 16,458 | 30% | $2.06 |

The whole agentic pipeline came to about 22,600 tokens and $0.079 per
paper. 37 papers (18.5%) needed revision, all of them papers on which the
offline reader swapped two numbers (the other 8 papers of that quarter had no
finding with two numbers to swap); supported claims rose from 96.3% of first
drafts to 100% of revised ones. No budget notice, nudge or `max_tokens`
retry was needed offline; the scripted tests cover those paths.

**Batch mode and cache lifetime.** On the same corpus with simulated
batch latency (300 s per batch, `ScaledClock`): online $17.33, batch with
5-minute cache entries $8.93 (52%), batch with 1-hour entries $11.40, online
with 1-hour entries $22.80. An agent's consecutive turns are minutes apart
in batch mode, so one would expect 1-hour entries to pay off, but their writes
cost 2x instead of 1.25x and most reads (the shared system prompt across
papers in a batch) still land within five minutes. `agents.cache_ttl`
therefore defaults to `5m`; `1h` is there for batch turnarounds that run
much longer than the simulated one.

To reproduce, run `Pipeline` over `synthetic_corpus(200)` with
`offline_client()` and a `Gateway` on a `ManualClock` (or a `ScaledClock`
for batch mode), once per strategy, and read the run statistics and
`Store.calls()`.

## 8. Offline mode

`precis run --offline` swaps the SDK client for a `FakeAnthropic` built from
every package's responders (`precis.offline.RESPONDERS`). The agent
responders recognise their role by the system prompt and act from the
request alone, as a model would: the reader reads three sections and submits
a draft built from their sentences, the critic reads the cited sections and
quotes the sentence that best matches each claim (`contradicted` when its
numbers differ or appear in another order), and the reviser re-reads failed
claims' sections and restates them. To exercise every correction path, the
reader swaps two numbers of a finding for one title in four (caught by the
critic and revised) and rounds a number in its first submission for another
one in four (rejected by validation and corrected). The responses carry no
usage; `FakeAnthropic` computes it from the prompt, modelling the prompt
cache. Offline runs use a `ScaledClock` running 600x, so batch mode's
queueing and polling take a fraction of a second.

## 9. Limitations

- The offline numbers above measure token flow and caching, not judgement;
  real reading depth, rejection and revision rates need a run against the
  API (`precis eval ablation` reports them per strategy).
- The token budget counts cache reads at full weight, so it limits context
  re-reads more than cost; the cost limit is `budget.max_cost_usd`.
- The number check exempts integers up to 12 and cannot see numbers the
  paper spells out ("twenty"); derived figures ("2x" from 10 vs 5) are
  rejected by design, which occasionally costs a turn for a legitimate
  restatement.
- The quote check accepts any quote that occurs somewhere in the paper,
  not necessarily in the cited section.
- Cancellation on SIGINT takes effect at the next point where a stage
  waits, so a stage whose last request has already returned may still be
  recorded `done`; only work that was actually cut off is left `pending`.
- `reset_running` assumes one process per store: a second concurrent run
  on the same store would reset the first one's running stages.
