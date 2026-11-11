"""The per-paper scheduler: read, verify and revise over thousands of papers.

Stage outputs follow the table in docs/design.md, section 1;
``precis.corpus.load`` reads them back as each paper's final summary.
"""

from precis.pipeline.progress import LiveProgress, render_progress
from precis.pipeline.scheduler import Pipeline
from precis.pipeline.stats import PIPELINE_STAGES, RunStats

__all__ = [
    "PIPELINE_STAGES",
    "LiveProgress",
    "Pipeline",
    "RunStats",
    "render_progress",
]
