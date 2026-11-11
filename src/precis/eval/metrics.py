"""Scoring helpers: ROUGE, bootstrap intervals and percentiles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel
from rouge_score import rouge_scorer

from precis.models import PaperSummary

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")
_SCORER = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)


class Rouge(BaseModel):
    """ROUGE F1 scores (0-1), with Porter stemming."""

    rouge1: float
    rouge2: float
    rougeL: float


def summary_text(summary: PaperSummary) -> str:
    """The parts of a summary that an abstract also covers, as one text."""
    return " ".join(
        [summary.tldr, summary.problem, summary.approach, *(f.claim for f in summary.findings)]
    )


def rouge(candidate: str, reference: str) -> Rouge:
    scores = _SCORER.score(reference, candidate)
    return Rouge(**{name: float(scores[name].fmeasure) for name in ROUGE_TYPES})


def mean_rouge(scores: Sequence[Rouge]) -> Rouge:
    if not scores:
        return Rouge(rouge1=0.0, rouge2=0.0, rougeL=0.0)
    return Rouge(
        **{name: float(np.mean([getattr(s, name) for s in scores])) for name in ROUGE_TYPES}
    )


class Estimate(BaseModel):
    """A mean with a 95% bootstrap confidence interval."""

    mean: float
    low: float
    high: float
    n: int


def bootstrap(values: Sequence[float], *, seed: int = 0, resamples: int = 2000) -> Estimate:
    """Mean of ``values`` with a percentile bootstrap 95% interval.

    With 50 papers the interval on key-point recall is typically +-5 points,
    which is the resolution at which strategies can be told apart.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return Estimate(mean=0.0, low=0.0, high=0.0, n=0)
    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(resamples, data.size), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return Estimate(mean=float(data.mean()), low=float(low), high=float(high), n=int(data.size))


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0-100), or 0 for no values."""
    return float(np.percentile(values, q)) if len(values) else 0.0
