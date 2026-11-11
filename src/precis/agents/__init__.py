"""LLM agents that read, verify and revise paper summaries.

``summarize`` writes a summary with the configured strategy (the ``agentic``
reader by default), ``verify_summary`` checks it claim by claim, and
``revise_summary`` rewrites the claims that failed. Each has a ``run_*``
twin that also returns the run's statistics (``AgentTrace``), which the
scheduler aggregates. All of them reach the model only through
``precis.llm.ModelGateway``.
"""

from precis.agents.critic import quotes_not_found, run_critic, verify_summary
from precis.agents.errors import AgentError, AgentRefused
from precis.agents.reader import read_paper, run_reader
from precis.agents.reviser import revise_summary, run_reviser
from precis.agents.strategies import run_summarize, summarize
from precis.agents.trace import AgentResult, AgentTrace

__all__ = [
    "AgentError",
    "AgentRefused",
    "AgentResult",
    "AgentTrace",
    "quotes_not_found",
    "read_paper",
    "revise_summary",
    "run_critic",
    "run_reader",
    "run_reviser",
    "run_summarize",
    "summarize",
    "verify_summary",
]
