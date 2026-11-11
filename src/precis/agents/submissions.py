"""Submit tools and the validation behind them.

A submission is parsed into its record and then checked for what a JSON
schema cannot express: word and list limits, cited ids that must exist in
the document, and numbers that must appear in the paper. A failed check is
returned to the model as an error ``tool_result`` listing every problem at
once, so one correction round usually suffices. Harmless differences
(keyword case, stray whitespace, duplicates) are fixed silently instead of
costing the model a turn.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from precis.agents.paper import PaperView, numbers_in
from precis.agents.schema import strict_schema
from precis.models import Finding, SummaryDraft

MAX_TLDR_WORDS = 40
FINDINGS_RANGE = (2, 6)
KEYWORDS_RANGE = (3, 8)


class SubmissionRejected(Exception):
    """A submission failed validation; the message is written for the model."""


@dataclass(frozen=True)
class SubmitTool[T]:
    """A tool whose valid call ends the agent's run with a result."""

    name: str
    description: str
    input_schema: dict[str, Any]
    parse: Callable[[Mapping[str, Any]], T]

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": self.input_schema,
        }


def describe_validation_error(error: ValidationError) -> list[str]:
    return [
        f"{'.'.join(str(part) for part in item['loc']) or 'input'}: {item['msg']}"
        for item in error.errors()
    ]


def rejection(problems: list[str], name: str) -> SubmissionRejected:
    listed = "\n".join(f"- {p}" for p in problems)
    return SubmissionRejected(
        f"The submission was not accepted:\n{listed}\nFix these and call {name} again "
        "with the complete submission."
    )


# -- summaries -------------------------------------------------------------

SUMMARY_DESCRIPTIONS = {
    "tldr": "One sentence, at most 40 words: what the paper does and its headline result.",
    "problem": "The problem and why it matters, in 1-3 sentences.",
    "approach": "How the paper attacks the problem, in 2-4 sentences.",
    "findings": "2-6 key results, most important first.",
    "findings.claim": "One specific sentence, with the paper's own numbers when it gives them.",
    "findings.section_ids": "Ids of the sections or captions that contain the evidence.",
    "contributions": "What the paper adds, one item each.",
    "limitations": "Limitations stated by the authors or evident from the setup.",
    "methods": "Named techniques, models, algorithms or tools.",
    "datasets": "Datasets, benchmarks or workloads evaluated on.",
    "keywords": "3-8 lowercase topic keywords.",
}

SUMMARY_SCHEMA = strict_schema(SummaryDraft, descriptions=SUMMARY_DESCRIPTIONS)

SUBMIT_SUMMARY_NAME = "submit_summary"

# Text fields whose numbers must appear in the paper. Names in methods,
# datasets and keywords ("ResNet-50") are left alone.
_NUMBER_CHECKED = ("tldr", "problem", "approach", "contributions", "limitations")


def summary_tool(view: PaperView) -> SubmitTool[SummaryDraft]:
    return SubmitTool(
        name=SUBMIT_SUMMARY_NAME,
        description=(
            "Submit the finished summary of the paper. Call it once, when you have read "
            "enough to support every claim in it; a valid submission ends your work. If it "
            "is rejected, fix what the error names and call it again with the complete summary."
        ),
        input_schema=SUMMARY_SCHEMA,
        parse=lambda data: parse_summary(view, data),
    )


def parse_summary(view: PaperView, data: Mapping[str, Any]) -> SummaryDraft:
    """Validate a summary against its record and against the paper."""
    try:
        draft = tidy_summary(SummaryDraft.model_validate(data))
    except ValidationError as e:
        raise rejection(describe_validation_error(e), SUBMIT_SUMMARY_NAME) from e
    problems = summary_problems(view, draft)
    if problems:
        raise rejection(problems, SUBMIT_SUMMARY_NAME)
    return draft


def tidy_summary(draft: SummaryDraft) -> SummaryDraft:
    """Strip whitespace, drop empty and duplicate list items, lowercase keywords."""

    def items(values: list[str], *, lower: bool = False) -> list[str]:
        cleaned = (v.strip().lower() if lower else v.strip() for v in values)
        return list(dict.fromkeys(v for v in cleaned if v))

    findings = [
        Finding(claim=f.claim.strip(), section_ids=items(f.section_ids))
        for f in draft.findings
        if f.claim.strip()
    ]
    return SummaryDraft(
        tldr=draft.tldr.strip(),
        problem=draft.problem.strip(),
        approach=draft.approach.strip(),
        findings=findings,
        contributions=items(draft.contributions),
        limitations=items(draft.limitations),
        methods=items(draft.methods),
        datasets=items(draft.datasets),
        keywords=items(draft.keywords, lower=True),
    )


def summary_problems(view: PaperView, draft: SummaryDraft) -> list[str]:
    problems = [
        f"{name} is empty." for name in ("tldr", "problem", "approach") if not getattr(draft, name)
    ]
    words = len(draft.tldr.split())
    if words > MAX_TLDR_WORDS:
        problems.append(f"tldr has {words} words; the limit is {MAX_TLDR_WORDS}.")
    low, high = FINDINGS_RANGE
    if not low <= len(draft.findings) <= high:
        problems.append(f"findings has {len(draft.findings)} items; give {low} to {high}.")
    for i, finding in enumerate(draft.findings):
        if not finding.section_ids:
            problems.append(f"finding {i} cites no section_ids.")
        unknown = [
            s for s in finding.section_ids if s not in view.sections and s not in view.captions
        ]
        if unknown:
            problems.append(
                f"finding {i} cites {', '.join(map(repr, unknown))}, which "
                f"{'is not an id' if len(unknown) == 1 else 'are not ids'} in this paper; "
                "use section ids from the outline or caption ids from read_captions."
            )
    low, high = KEYWORDS_RANGE
    if not low <= len(draft.keywords) <= high:
        problems.append(f"keywords has {len(draft.keywords)} items; give {low} to {high}.")
    problems += _invented_numbers(view, draft)
    return problems


def _invented_numbers(view: PaperView, draft: SummaryDraft) -> list[str]:
    fields: list[tuple[str, str]] = []
    for name in _NUMBER_CHECKED:
        value = getattr(draft, name)
        fields += [(name, v) for v in value] if isinstance(value, list) else [(name, value)]
    fields += [(f"finding {i}", f.claim) for i, f in enumerate(draft.findings)]
    problems = []
    for name, text in fields:
        missing = [n for n in dict.fromkeys(numbers_in(text)) if n not in view.numbers]
        if missing:
            problems.append(
                f"{name} states {', '.join(missing)}, which "
                f"{'does' if len(missing) == 1 else 'do'} not appear anywhere in the paper. "
                "Use the paper's exact figures, or state the result without a number."
            )
    return problems
