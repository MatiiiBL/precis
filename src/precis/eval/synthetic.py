"""A synthetic corpus with the shape of real papers, for the offline benchmarks.

Throughput depends on how much text the agents read and write, not on what
it says, so the benchmark papers only need realistic structure and size:
an abstract, an introduction that states a contribution, a method with
subsections, results with numbers, a discussion with limitations, figure
and table captions, and a body length drawn from a log-normal distribution
(median about 8,500 tokens, 95th percentile about 17,500), close to arXiv
papers in computer architecture once the bibliography is dropped.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

from precis.models import Caption, Document, PaperMeta, Section
from precis.util import estimate_tokens, sha256_hex

_SUBJECTS = [
    ("cache replacement", "last-level caches", "hit rate", "LRU"),
    ("hardware prefetching", "memory-intensive workloads", "coverage", "a stride prefetcher"),
    ("systolic arrays", "DNN inference", "throughput", "a weight-stationary baseline"),
    ("sparse tensor accelerators", "pruned transformers", "energy efficiency", "a dense design"),
    ("network-on-chip routing", "chiplet systems", "tail latency", "XY routing"),
    ("qubit mapping", "superconducting devices", "gate count", "SABRE"),
    ("processing-in-memory", "graph analytics", "speedup", "a GPU baseline"),
    ("branch prediction", "server workloads", "MPKI", "TAGE-SC-L"),
    ("DRAM refresh", "high-density memory", "bandwidth", "auto-refresh"),
    ("secure enclaves", "cloud tenants", "overhead", "SGX"),
]
_METHODS = [
    "a learned policy",
    "a lightweight predictor",
    "a co-design",
    "a compiler pass",
    "a dataflow",
    "a scheduling algorithm",
    "a hardware table",
    "an analytical model",
]
_BENCHMARKS = ["SPEC CPU2017", "GAP", "MLPerf", "PARSEC", "CloudSuite", "QASMBench"]
_FILLER = [
    "This matters because {area} dominate the cost of modern systems.",
    "Prior work on {subject} relies on fixed heuristics that ignore program behavior.",
    "We model the {metric} of {area} as a function of the access stream.",
    "The design keeps the critical path unchanged and adds a small amount of state.",
    "Each component is evaluated in isolation before the full system is measured.",
    "The configuration follows the parameters of a recent commercial processor.",
    "We report the geometric mean across all workloads unless stated otherwise.",
    "Sensitivity to table size and associativity is discussed in the appendix.",
    "Our implementation is open source and integrates with an existing simulator.",
]


def _sentence(rng: random.Random, subject: tuple[str, str, str, str]) -> str:
    text = rng.choice(_FILLER)
    return text.format(subject=subject[0], area=subject[1], metric=subject[2])


def _paragraphs(rng: random.Random, subject: tuple[str, str, str, str], tokens: int) -> str:
    paragraphs, used = [], 0
    while used < tokens:
        paragraph = " ".join(_sentence(rng, subject) for _ in range(rng.randint(4, 7)))
        paragraphs.append(paragraph)
        used += estimate_tokens(paragraph)
    return "\n\n".join(paragraphs)


def _section(id_: str, title: str, level: int, text: str) -> Section:
    return Section(id=id_, title=title, level=level, text=text, n_tokens=estimate_tokens(text))


def synthetic_paper(index: int, *, seed: int = 0) -> tuple[PaperMeta, Document]:
    """Paper ``index`` of the synthetic corpus (the same paper for the same seed)."""
    rng = random.Random(f"{seed}:{index}")
    subject = _SUBJECTS[index % len(_SUBJECTS)]
    method = rng.choice(_METHODS)
    benchmark = rng.choice(_BENCHMARKS)
    gain = round(rng.uniform(4, 40), 1)
    cost = round(rng.uniform(0.5, 6), 1)
    name = f"{subject[0].split()[0].capitalize()}{rng.randint(10, 99)}"
    body_tokens = int(9000 * math.exp(rng.gauss(0, 0.45)))
    share = {
        "intro": 0.11,
        "background": 0.11,
        "method": 0.15,
        "overview": 0.15,
        "results": 0.28,
        "discussion": 0.12,
    }

    def fill(part: str) -> str:
        # Paragraphs overshoot their target by half a paragraph on average.
        return _paragraphs(rng, subject, int(body_tokens * share[part]) - 60)

    contribution = (
        f"We propose {name}, {method} for {subject[0]} in {subject[1]}. "
        f"{name} improves {subject[2]} by {gain}% over {subject[3]} on {benchmark}."
    )
    result = (
        f"On {benchmark}, {name} improves {subject[2]} by {gain}% over {subject[3]}, "
        f"with a storage overhead of {cost} KB per core."
    )
    limitation = (
        f"However, {name} is evaluated only in simulation and does not model multi-socket systems."
    )
    sections = [
        _section("s1", "Introduction", 1, contribution + "\n\n" + fill("intro")),
        _section("s2", "Background and Related Work", 1, fill("background")),
        _section("s3", f"The {name} Design", 1, fill("method")),
        _section("s3.1", "Overview", 2, f"{name} consists of two parts.\n\n" + fill("overview")),
        _section("s4", "Evaluation", 1, result + "\n\n" + fill("results")),
        _section("s5", "Discussion and Limitations", 1, limitation + "\n\n" + fill("discussion")),
        _section("s6", "Conclusion", 1, contribution),
    ]
    captions = [
        Caption(id="fig1", kind="figure", text=f"Figure 1: Overview of {name}."),
        Caption(id="tab1", kind="table", text=f"Table 1: {subject[2]} on {benchmark}."),
    ]
    paper_id = f"2601.{index:05d}"
    abstract = (
        f"{subject[1].capitalize()} are limited by {subject[0]}. {contribution} "
        f"{name} needs only {cost} KB of state per core."
    )
    meta = PaperMeta(
        id=paper_id,
        version=1,
        title=f"{name}: {method.removeprefix('a ').removeprefix('an ').capitalize()} "
        f"for {subject[0].title()}",
        authors=[f"Author {rng.randint(1, 500)}", f"Author {rng.randint(1, 500)}"],
        abstract=abstract,
        categories=["cs.AR"],
        primary_category="cs.AR",
        published=date(2026, 1, 1) + timedelta(days=index % 180),
        url=f"https://arxiv.org/abs/{paper_id}",
    )
    doc = Document(
        paper_id=paper_id,
        source_format="html",
        sections=sections,
        captions=captions,
        n_tokens=sum(s.n_tokens for s in sections),
        sha256=sha256_hex(paper_id + str(seed)),
    )
    return meta, doc


def synthetic_corpus(n: int, *, seed: int = 0) -> list[tuple[PaperMeta, Document]]:
    return [synthetic_paper(i, seed=seed) for i in range(n)]
