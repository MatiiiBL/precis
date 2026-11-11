"""Run configuration, loaded from a TOML file (``precis.toml`` by default).

Every field has a default, so an empty file (or none) is a valid
configuration. Each section belongs to one package; see docs/design.md.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field

from precis import __version__
from precis.models import Record

DEFAULT_MODEL = "claude-opus-5-5"

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class AgentConfig(Record):
    """Model settings for one LLM role."""

    model: str = DEFAULT_MODEL
    effort: Effort = "medium"
    max_tokens: int = Field(default=16000, gt=0, description="Per response.")
    max_turns: int = Field(
        default=10, gt=0, description="Tool-use turns before the agent must answer."
    )
    token_budget: int = Field(
        default=80000,
        gt=0,
        description="Tokens one agent run may spend before it is told to finish: input, cache "
        "writes and output, plus cache reads at a tenth.",
    )


CacheTTL = Literal["5m", "1h"]


class AgentsConfig(Record):
    reader: AgentConfig = AgentConfig()
    critic: AgentConfig = AgentConfig()
    reviser: AgentConfig = AgentConfig()
    synthesizer: AgentConfig = AgentConfig(effort="high", max_turns=1)
    judge: AgentConfig = AgentConfig(effort="low", max_turns=1)
    strategy: Literal["agentic", "fulltext", "mapreduce"] = "agentic"
    include_abstract: bool = Field(
        default=True, description="Show the author abstract to the reader (off for evaluation)."
    )
    revise: bool = Field(default=True, description="Rewrite summaries that fail verification once.")
    page_tokens: int = Field(
        default=2500, gt=0, description="Size of one read_section page, in estimated tokens."
    )
    search_results: int = Field(default=5, gt=0, description="Passages returned by search_paper.")
    chunk_tokens: int = Field(
        default=8000, gt=0, description="Body tokens per note-taking request (mapreduce)."
    )
    cache_ttl: CacheTTL = Field(
        default="5m",
        description="Prompt-cache lifetime of agent requests; 1h writes cost 2x instead of 1.25x "
        "but survive longer gaps between turns.",
    )
    cache_turns: bool = Field(
        default=True,
        description="Cache each turn of a tool loop for the next one (automatic caching). Off in "
        "batch mode, where turns are minutes apart; the system prompt is cached either way.",
    )

    def roles(self) -> dict[str, AgentConfig]:
        """Every model role, by name, so checks over all roles cannot miss a new one."""
        return {name: value for name, value in self if isinstance(value, AgentConfig)}


class LimitsConfig(Record):
    """Client-side admission control, retries and batching for the Messages API."""

    max_concurrency: int = Field(default=64, gt=0, description="Ceiling for adaptive concurrency.")
    initial_concurrency: int = Field(default=8, gt=0)
    slow_start: bool = Field(
        default=True,
        description="Double the concurrency limit per window of successes until the first "
        "429/529 or the ceiling, then grow it by one per window.",
    )
    rpm: int | None = Field(default=None, description="Requests/min; None = learn from headers.")
    itpm: int | None = Field(default=None, description="Input tokens/min; None = learn.")
    otpm: int | None = Field(default=None, description="Output tokens/min; None = learn.")
    burst_s: float = Field(
        default=10.0,
        gt=0,
        description="Seconds of each per-minute limit that may be sent in one burst.",
    )
    max_attempts: int = Field(default=8, gt=0)
    request_timeout_s: float = Field(
        default=600.0,
        gt=0,
        description="Seconds a streamed request may go without data before it times out.",
    )
    backoff_base_s: float = Field(
        default=1.0, gt=0, description="First retry waits up to this long (full jitter)."
    )
    backoff_max_s: float = Field(default=60.0, gt=0, description="Cap on one retry's wait.")
    batch_max_requests: int = Field(
        default=500, gt=0, le=100_000, description="Batch mode: send a batch at this many requests."
    )
    batch_max_bytes: int = Field(
        default=200_000_000,
        gt=0,
        description="Batch mode: send a batch before its requests reach this size (API: 256 MB).",
    )
    batch_max_wait_s: float = Field(
        default=30.0,
        gt=0,
        description="Batch mode: send a batch this long after its first request.",
    )
    batch_poll_s: float = Field(
        default=10.0, gt=0, description="First wait before polling a batch."
    )
    batch_poll_max_s: float = Field(default=120.0, gt=0, description="Cap on the polling interval.")


class BudgetConfig(Record):
    max_cost_usd: float | None = Field(
        default=None,
        description=(
            "Cap on the total spend recorded in the store, earlier runs included: a request "
            "is refused when its reservation would cross it. Replies longer than their "
            "reservation can overshoot by up to about max_concurrency x max_tokens x the "
            "output price. None = no cap."
        ),
    )


class FetchConfig(Record):
    min_interval_s: float = Field(
        default=3.0, ge=0, description="Gap between requests to arXiv (their published guidance)."
    )
    user_agent: str = Field(
        default=f"precis/{__version__}",
        description=(
            "Sent with every arXiv request. Add a contact so arXiv can reach a heavy "
            'client instead of blocking it: "precis/0.1.0 (mailto:you@example.org)".'
        ),
    )
    prefer: Literal["html", "pdf"] = "html"
    raw_dir: Path = Path("data/raw")
    page_size: int = Field(
        default=200, gt=0, le=2000, description="Results per arXiv API request (API max 2000)."
    )
    max_attempts: int = Field(
        default=5, gt=0, description="Tries per request on 429, 5xx, timeouts and empty pages."
    )
    timeout_s: float = Field(default=60.0, gt=0, description="Per-request timeout.")


class PipelineConfig(Record):
    max_papers_in_flight: int = Field(default=64, gt=0)
    use_cache: bool = Field(
        default=True, description="Reuse stored responses for identical requests."
    )
    fallbacks: bool = Field(
        default=True, description="Ask the API to retry refused requests on a fallback model."
    )


class CorpusConfig(Record):
    """Topic modelling and report synthesis over the finished summaries."""

    seed: int = Field(default=0, description="Seeds SVD and k-means, so topics are reproducible.")
    k: int | None = Field(default=None, gt=0, description="Number of topics; None = silhouette.")
    k_min: int = Field(default=6, ge=2, description="Smallest k tried by the silhouette search.")
    k_max: int = Field(
        default=24, ge=2, description="Largest k tried by the silhouette search (and ~sqrt(n))."
    )
    min_topic_size: int = Field(
        default=3, ge=1, description="The silhouette search avoids k with a topic this small."
    )
    svd_components: int = Field(default=100, gt=1, description="Dimensions kept by truncated SVD.")
    central_papers: int = Field(
        default=8, gt=0, description="Most central papers shown to the model that names a topic."
    )
    synthesis_input_tokens: int = Field(
        default=60000,
        ge=2000,
        description="Most paper tokens in one synthesis request; larger topics are summarized "
        "hierarchically.",
    )


class BenchConfig(Record):
    """The simulated API that the offline throughput and fault benchmarks run against."""

    papers: int = Field(default=2000, gt=0, description="Papers in the throughput benchmark.")
    concurrency: list[int] = Field(
        default=[4, 8, 16, 32, 64, 128, 256], description="Concurrency ceilings to sweep."
    )
    rpm: int = Field(default=4000, gt=0, description="Simulated server limit, requests/min.")
    itpm: int = Field(default=2_000_000, gt=0, description="Simulated input tokens/min.")
    otpm: int = Field(default=400_000, gt=0, description="Simulated output tokens/min.")
    faults_papers: int = Field(default=500, gt=0, description="Papers in the fault benchmark.")
    faults_concurrency: int = Field(default=32, gt=0)
    seed: int = 0


class EvalConfig(Record):
    """Quality evaluations and benchmarks."""

    n: int = Field(default=50, gt=0, description="Papers sampled by eval abstract and ablation.")
    seed: int = Field(default=0, description="Seeds the paper sample.")
    results_dir: Path = Path("results")
    figures_dir: Path = Path("docs/img")
    bench: BenchConfig = BenchConfig()


class Config(Record):
    store_path: Path = Path("data/precis.db")
    agents: AgentsConfig = AgentsConfig()
    limits: LimitsConfig = LimitsConfig()
    budget: BudgetConfig = BudgetConfig()
    fetch: FetchConfig = FetchConfig()
    pipeline: PipelineConfig = PipelineConfig()
    corpus: CorpusConfig = CorpusConfig()
    eval: EvalConfig = EvalConfig()


def load_config(path: Path | None = None) -> Config:
    """Load ``path``, or ``precis.toml`` in the working directory if it exists."""
    if path is None:
        default = Path("precis.toml")
        if not default.exists():
            return Config()
        path = default
    with path.open("rb") as f:
        return Config.model_validate(tomllib.load(f))
