"""Synthetic papers and an offline pipeline for the scheduler tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from precis.config import Config
from precis.llm import Gateway, ManualClock
from precis.llm.fake import FakeAnthropic, Responder
from precis.models import Document, PaperMeta, Section, Stage, StageRecord, StageStatus
from precis.offline import RESPONDERS
from precis.pipeline import Pipeline
from precis.store import Store
from precis.util import estimate_tokens

TOPICS = ["cache replacement", "branch prediction", "memory compression", "network-on-chip routing"]


def paper(i: int, title: str | None = None) -> tuple[PaperMeta, Document]:
    """A small, distinct paper; its numbers depend on ``i``."""
    topic = TOPICS[i % len(TOPICS)]
    gain, cost = 14.5 + 3.5 * i, 20 + i
    texts = {
        "Introduction": (
            f"Hardware for {topic} is a bottleneck in modern processors. "
            f"We propose Design{i}, a new scheme for {topic}."
        ),
        "Method": (
            f"Design{i} keeps a table of {256 * (i + 1)} entries indexed by address. "
            f"Each entry stores a {cost}-bit confidence counter."
        ),
        "Results": (
            f"Design{i} improves performance by {gain}% over the baseline on SPEC CPU2017, "
            f"at a storage cost of {cost} KB. It needs no changes to the instruction set."
        ),
        "Conclusion": f"Design{i} is practical. However, it was evaluated only in simulation.",
    }
    sections = [
        Section(id=f"s{n}", title=t, level=1, text=x, n_tokens=estimate_tokens(x))
        for n, (t, x) in enumerate(texts.items(), 1)
    ]
    meta = PaperMeta(
        id=f"2501.{i:05d}",
        title=title or f"Design{i}: Faster {topic.title()} for Processors",
        authors=["A. Author"],
        abstract=f"We present Design{i} for {topic}.",
        categories=["cs.AR"],
        primary_category="cs.AR",
        published=date(2025, 1, 1 + i),
        url=f"https://arxiv.org/abs/2501.{i:05d}",
    )
    doc = Document(
        paper_id=meta.id,
        source_format="html",
        sections=sections,
        captions=[],
        n_tokens=sum(s.n_tokens for s in sections),
        sha256=f"{i:064d}",
    )
    return meta, doc


def add_papers(store: Store, n: int, **titles: str) -> list[str]:
    """Store ``n`` papers with their documents and a done fetch stage, as precis fetch does."""
    ids = []
    for i in range(n):
        meta, doc = paper(i, titles.get(f"p{i}"))
        store.upsert_paper(meta)
        store.put_document(doc)
        store.set_stage(
            StageRecord(
                paper_id=meta.id,
                stage=Stage.FETCH,
                status=StageStatus.DONE,
                attempts=1,
                config_hash="fetch",
                run_id="fetch",
                updated_at=datetime.now(UTC),
            )
        )
        ids.append(meta.id)
    return ids


def config(**sections: dict[str, Any]) -> Config:
    base: dict[str, Any] = {"pipeline": {"fallbacks": False}}
    for name, values in sections.items():
        base[name] = {**base.get(name, {}), **values}
    return Config.model_validate(base)


@dataclass
class Setup:
    store: Store
    fake: FakeAnthropic
    config: Config

    def pipeline(self, run_id: str, *, config: Config | None = None) -> Pipeline:
        cfg = config or self.config
        clock = ManualClock()
        gateway = Gateway(self.fake, self.store, run_id=run_id, config=cfg, clock=clock)
        return Pipeline(cfg, self.store, gateway)


def setup(*extra: Responder, cfg: Config | None = None) -> Setup:
    """An in-memory store and an offline client; ``extra`` responders take precedence."""
    fake = FakeAnthropic([*extra, *RESPONDERS], clock=ManualClock(), seed=1)
    return Setup(Store(":memory:"), fake, cfg or config())


def brief_title(params: dict[str, Any]) -> str:
    match = re.search(r"^Title: (.*)$", params["messages"][0]["content"], re.MULTILINE)
    return match.group(1) if match else ""
