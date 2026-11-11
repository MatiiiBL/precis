"""Synthetic papers, summaries and a scripted model for corpus and eval tests."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from anthropic.types import Message
from pydantic import BaseModel

from precis.corpus.load import FinishedPaper
from precis.llm.base import Reply
from precis.models import (
    ClaimCheck,
    Finding,
    PaperMeta,
    PaperSummary,
    RevisionResult,
    Stage,
    StageRecord,
    StageStatus,
    Verdict,
    Verification,
)
from precis.store import Store

# Each theme has its own vocabulary; papers draw most of their words from
# their theme, so a working clustering recovers the themes.
THEMES: dict[str, list[str]] = {
    "caches": [
        "cache replacement",
        "prefetching",
        "last-level cache",
        "hit rate",
        "memory hierarchy",
        "cache partitioning",
        "eviction policy",
        "miss latency",
    ],
    "accelerators": [
        "systolic array",
        "dnn accelerator",
        "dataflow",
        "sparsity",
        "processing elements",
        "tensor cores",
        "quantized inference",
        "weight stationary",
    ],
    "quantum": [
        "qubit routing",
        "quantum circuits",
        "error correction",
        "surface code",
        "gate fidelity",
        "transpiler",
        "superconducting qubits",
        "decoherence",
    ],
    "networks": [
        "network-on-chip",
        "router microarchitecture",
        "flow control",
        "chiplet interconnect",
        "deadlock freedom",
        "link bandwidth",
        "mesh topology",
        "packet switching",
    ],
}
GENERIC = ["simulation", "energy", "latency", "throughput", "area", "benchmark"]


def make_meta(paper_id: str, *, title: str | None = None, abstract: str = "") -> PaperMeta:
    return PaperMeta(
        id=paper_id,
        version=1,
        title=title or f"Paper {paper_id}",
        authors=["Ada Lovelace", "Alan Turing"],
        abstract=abstract or f"We study {paper_id}.",
        categories=["cs.AR"],
        primary_category="cs.AR",
        published=date(2025, 7, 1),
        url=f"https://arxiv.org/abs/{paper_id}",
    )


def make_summary(
    paper_id: str, words: list[str], *, revision: int = 0, n_findings: int = 2
) -> PaperSummary:
    phrase = ", ".join(words)
    return PaperSummary(
        paper_id=paper_id,
        tldr=f"A study of {phrase}.",
        problem=f"Systems suffer when {words[0]} is poor.",
        approach=f"The authors combine {phrase} in one design.",
        findings=[
            Finding(claim=f"{words[i % len(words)]} improves by {10 + i}%.", section_ids=["s4"])
            for i in range(n_findings)
        ],
        contributions=[f"A new {words[0]} scheme."],
        limitations=["Evaluated in simulation only."],
        methods=words[:2],
        datasets=["SPEC CPU2017"],
        keywords=words[:4],
        revision=revision,
    )


def make_verification(
    summary: PaperSummary, verdicts: list[Verdict] | None = None, *, evidence: str = "quote"
) -> Verification:
    targets = ["tldr"] + [f"finding:{i}" for i in range(len(summary.findings))]
    verdicts = verdicts or [Verdict.SUPPORTED] * len(targets)
    return Verification(
        paper_id=summary.paper_id,
        revision=summary.revision,
        checks=[
            ClaimCheck(target=t, verdict=v, evidence=evidence, section_id="s4", note="Checked.")
            for t, v in zip(targets, verdicts, strict=True)
        ],
    )


def make_view(
    paper_id: str,
    words: list[str],
    *,
    draft_verdicts: list[Verdict] | None = None,
    revised: bool = False,
    title: str | None = None,
) -> FinishedPaper:
    meta = make_meta(paper_id, title=title)
    draft = make_summary(paper_id, words)
    draft_check = make_verification(draft, draft_verdicts)
    if not revised:
        return FinishedPaper(meta, draft, draft_check, draft, draft_check)
    final = make_summary(paper_id, words, revision=1)
    return FinishedPaper(meta, final, make_verification(final), draft, draft_check)


def stage_record(
    paper_id: str,
    stage: Stage,
    output: BaseModel | None,
    status: StageStatus = StageStatus.DONE,
) -> StageRecord:
    return StageRecord(
        paper_id=paper_id,
        stage=stage,
        status=status,
        attempts=1,
        config_hash="h",
        run_id="run-1",
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        output=None if output is None else output.model_dump(mode="json"),
    )


def fill_store(store: Store, papers: list[FinishedPaper]) -> None:
    """Write papers the way the pipeline does: read, verify and revise records."""
    for paper in papers:
        store.upsert_paper(paper.meta)
        store.set_stage(stage_record(paper.id, Stage.READ, paper.draft))
        store.set_stage(stage_record(paper.id, Stage.VERIFY, paper.draft_verification))
        if paper.revised and paper.verification is not None:
            result = RevisionResult(summary=paper.summary, verification=paper.verification)
            store.set_stage(stage_record(paper.id, Stage.REVISE, result))
        else:
            store.set_stage(stage_record(paper.id, Stage.REVISE, None, StageStatus.SKIPPED))


def themed_corpus(per_theme: int, *, seed: int = 0) -> tuple[list[FinishedPaper], dict[str, str]]:
    """Papers drawn from ``THEMES``; also returns each paper's true theme."""
    rng = random.Random(seed)
    papers: list[FinishedPaper] = []
    truth: dict[str, str] = {}
    for t, (theme, vocabulary) in enumerate(sorted(THEMES.items())):
        for i in range(per_theme):
            paper_id = f"2507.{t:01d}{i:04d}"
            words = rng.sample(vocabulary, 4) + rng.sample(GENERIC, 1)
            papers.append(make_view(paper_id, words, title=f"{theme.title()} paper {i}"))
            truth[paper_id] = theme
    return papers, truth


def message(text: str, *, stop_reason: str = "end_turn", model: str = "claude-opus-5-5") -> Message:
    return Message.model_validate(
        {
            "id": "msg_scripted",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": text},
            ],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
    )


type Script = Callable[[dict[str, Any], str], Any]


@dataclass
class Call:
    params: dict[str, Any]
    stage: str
    paper_id: str | None

    @property
    def user_text(self) -> str:
        content = self.params["messages"][0]["content"]
        return content if isinstance(content, str) else json.dumps(content)


@dataclass
class ScriptedGateway:
    """A ``ModelGateway`` whose replies come from a function of the request.

    The script returns a JSON-serializable object (sent back as the reply's
    text) or a ready ``Message``.
    """

    script: Script
    calls: list[Call] = field(default_factory=list)

    async def create(
        self, params: dict[str, Any], *, stage: str, paper_id: str | None = None
    ) -> Reply:
        self.calls.append(Call(params, stage, paper_id))
        answer = self.script(params, stage)
        if isinstance(answer, Message):
            return answer
        return message(json.dumps(answer))

    def stages(self) -> list[str]:
        return [c.stage for c in self.calls]
