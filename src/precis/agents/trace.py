"""Per-run statistics an agent or strategy reports alongside its result."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from precis.llm import Reply


@dataclass
class AgentTrace:
    """What one agent run cost and did; the scheduler aggregates these per run.

    Token counts add up every request of the run, including cache reads, so a
    later turn that re-reads the conversation from cache counts it again: this
    is the quantity ``AgentConfig.token_budget`` limits.
    """

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    tool_calls: Counter[str] = field(default_factory=Counter)
    tool_errors: int = 0
    rejected_submissions: int = 0
    nudges: int = 0
    budget_notices: int = 0
    max_tokens_retries: int = 0
    quotes_not_found: int = 0
    tokens_read: int = 0
    fraction_read: float = 0.0

    @property
    def tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def add(self, reply: Reply) -> None:
        usage = reply.usage
        self.requests += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_creation_input_tokens += usage.cache_creation_input_tokens or 0
        self.cache_read_input_tokens += usage.cache_read_input_tokens or 0


@dataclass(frozen=True)
class AgentResult[T]:
    value: T
    trace: AgentTrace
