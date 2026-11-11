"""Offline stand-ins for the corpus requests (naming, synthesis, introduction).

Each responder recognizes its request by the system prompt and writes a
deterministic answer from the prompt alone, so ``precis topics``,
``precis report`` and ``precis site`` run end to end with ``--offline``.
The text is assembled from the summaries rather than written, but it has
the shape, citations and length of the real thing.
"""

from __future__ import annotations

import json
import re
from typing import Any

from precis.corpus.synthesis import (
    INTRO_SYSTEM,
    MERGE_SYSTEM,
    SYNTHESIS_SYSTEM,
    cited_ids,
)
from precis.corpus.topics import LABEL_SYSTEM
from precis.llm.fake import FakeReply, Responder


def _user(params: dict[str, Any]) -> str:
    content = params["messages"][0]["content"]
    return content if isinstance(content, str) else ""


def _field(prompt: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}: (.*)$", prompt, flags=re.M)
    return match.group(1).strip() if match else ""


def _json(answer: dict[str, Any]) -> FakeReply:
    return FakeReply.text(json.dumps(answer, ensure_ascii=False))


def label(params: dict[str, Any]) -> FakeReply | None:
    if params.get("system") != LABEL_SYSTEM:
        return None
    keywords = [k for k in _field(_user(params), "Distinctive keywords").split(", ") if k]
    name = " and ".join(k.title() for k in keywords[:2]) or "Miscellaneous"
    listed = ", ".join(keywords[:4]) or "a mix of subjects"
    return _json(
        {
            "label": name,
            "description": f"Papers on {listed}. They share methods and evaluation settings.",
        }
    )


def _sentences(prompt: str, prefix: str) -> list[str]:
    return [
        line.removeprefix(prefix).strip() for line in prompt.splitlines() if line.startswith(prefix)
    ]


def synthesize(params: dict[str, Any]) -> FakeReply | None:
    if params.get("system") not in (SYNTHESIS_SYSTEM, MERGE_SYSTEM):
        return None
    prompt = _user(params)
    topic = _field(prompt, "Topic")
    ids = list(dict.fromkeys(cited_ids(prompt)))
    if params["system"] == SYNTHESIS_SYSTEM:
        tldrs = _sentences(prompt, "TL;DR: ")
        findings = _sentences(prompt, "- ")
        claims = [f"{t.rstrip('.')} [arXiv:{pid}]." for t, pid in zip(tldrs, ids, strict=False)]
    else:
        drafts = re.findall(r"^Draft \d+:\n(.+)$", prompt, flags=re.M)
        claims = [d.split(". ")[0].rstrip(".") + "." for d in drafts]
        findings = _sentences(prompt, "- ")
    overview = (
        f"This section covers {len(ids)} papers on {topic.lower() or 'this topic'}. "
        + " ".join(claims[:4])
        + "\n\n"
        + " ".join(claims[4:8])
    ).strip()
    return _json(
        {
            "overview": overview,
            "trends": findings[:3] or [f"Work on {topic.lower()} is growing."],
            "open_problems": [
                "Most results come from simulation and remain to be confirmed on real systems."
            ],
            "key_papers": ids[:3],
        }
    )


def introduce(params: dict[str, Any]) -> FakeReply | None:
    if params.get("system") != INTRO_SYSTEM:
        return None
    prompt = _user(params)
    papers = _field(prompt, "Papers") or "several"
    topics = re.findall(r"^\d+\. (.+) \((\d+ papers?)\)$", prompt, flags=re.M)
    listed = ", ".join(f"{name} ({size})" for name, size in topics)
    category = _field(prompt, "Primary categories").split(" (")[0] or "research"
    return _json(
        {
            "title": f"Recent Work in {category}: {papers} Papers in {len(topics)} Topics",
            "introduction": f"This report surveys {papers} recent papers in {len(topics)} "
            f"topics: {listed}.\n\nEach section below synthesizes one topic.",
        }
    )


RESPONDERS: tuple[Responder, ...] = (label, synthesize, introduce)
