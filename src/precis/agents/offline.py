"""Deterministic stand-ins for the model, for ``--offline`` runs and tests.

Each responder plays one role from the request alone, the way the model
would: it recognizes its role by the system prompt, reads the tool results
already in the conversation, and either calls tools or submits. The reader
opens the introduction, the method and the results and builds its summary
from their sentences; the critic reads the cited sections and quotes the
sentence that best matches each claim; the reviser re-reads the failed
claims' sections and replaces them with what the paper says. For one paper
in four (by title hash) the reader misattributes a number in its last
finding, so offline runs exercise verification and revision too.

The responses are real ``anthropic.types.Message`` objects (with zero usage,
which ``FakeAnthropic`` replaces with its prompt-cache model), so the
gateway, the accounting and the agents run unchanged. Nothing here is a
model of quality; it exists so every command runs without an API key.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from anthropic.types import Message

from precis.agents.critic import SUBMIT_VERIFICATION_NAME
from precis.agents.paper import numbers_in, split_sentences
from precis.agents.prompts import PromptName, load_prompt
from precis.agents.submissions import MAX_TLDR_WORDS, SUBMIT_SUMMARY_NAME
from precis.util import canonical_json, sha256_hex

OfflineResponder = Callable[[dict[str, Any]], Message | None]

_ID = r"[a-z]+\d+(?:\.\d+)*"
_OUTLINE_LINE = re.compile(rf"^\s*({_ID})  (.+?)  \((.+)\)$", re.MULTILINE)
_CLAIM_LINE = re.compile(r"^\[(tldr|finding:\d+)\] (.*?)(?: \(cites: (.*)\))?$", re.MULTILINE)
_CHECK_LINE = re.compile(r"^\[(tldr|finding:\d+)\] (\w+):", re.MULTILINE)
_BLOCK_START = re.compile(rf"\n\n(?=\[{_ID}\] )")
_HEADED_BLOCK = re.compile(rf"^\[({_ID})\] (.*?)(?:\n\n|$)(.*)", re.DOTALL)
_NOTE_LINE = re.compile(rf"^\[({_ID})\] (.+)$", re.MULTILINE)
_SEARCH_HIT = re.compile(rf"^\d+\. \[({_ID})\][^\n]*\n(.+)", re.DOTALL)
_WORD = re.compile(r"[a-z0-9]+(?:\.\d+)?")

_INTRO = re.compile(r"intro|overview|motivation", re.IGNORECASE)
_METHOD = re.compile(r"method|approach|design|architecture|model|system|framework", re.IGNORECASE)
_RESULTS = re.compile(r"result|evaluation|experiment|benchmark", re.IGNORECASE)
_CONTRIBUTION = re.compile(
    r"\bwe (?:propose|present|introduce|design|show)|contribut", re.IGNORECASE
)
_LIMITATION = re.compile(r"limitation|however|future work|only\b|does not", re.IGNORECASE)
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9-]{1,}[A-Z0-9]\b")
# Words of five or more letters that make poor keywords.
_STOPWORDS = frozenset(
    {
        "about", "above", "after", "again", "against", "among", "based", "because",
        "before", "being", "below", "between", "could", "doing", "during", "further",
        "having", "itself", "other", "should", "their", "there", "these", "those",
        "through", "under", "until", "where", "which", "while", "would", "paper",
        "results", "using", "shows", "shown", "table", "figure", "section", "method",
        "methods", "approach",
    }
)  # fmt: skip

_SECTIONS_TO_READ = 3


@dataclass(frozen=True)
class _Passage:
    section_id: str
    title: str
    text: str


@dataclass(frozen=True)
class _Exchange:
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool


# -- request parsing ---------------------------------------------------------


def _role_is(params: dict[str, Any], name: PromptName) -> bool:
    system = params.get("system")
    return (
        isinstance(system, list)
        and bool(system)
        and system[0].get("text") == load_prompt(name).text
    )


def _tagged(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""


def _brief(params: dict[str, Any]) -> str:
    content = params["messages"][0]["content"]
    return content if isinstance(content, str) else ""


def _exchanges(messages: Iterable[dict[str, Any]]) -> list[_Exchange]:
    calls: dict[str, dict[str, Any]] = {}
    out = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            continue
        for block in content:
            if block.get("type") == "tool_use":
                calls[block["id"]] = block
            elif block.get("type") == "tool_result" and block["tool_use_id"] in calls:
                call = calls[block["tool_use_id"]]
                out.append(
                    _Exchange(
                        call["name"],
                        call["input"],
                        str(block["content"]),
                        bool(block.get("is_error")),
                    )
                )
    return out


def _must_finish(messages: list[dict[str, Any]]) -> bool:
    """A budget notice or a nudge is in the conversation."""
    return any(m["role"] == "system" for m in messages) or (
        len(messages) > 1
        and messages[-1]["role"] == "user"
        and isinstance(messages[-1]["content"], str)
    )


def _read_passages(exchanges: Iterable[_Exchange]) -> list[_Passage]:
    """The text the agent has read, from read_section and search_paper results."""
    passages = []
    for ex in exchanges:
        if ex.is_error:
            continue
        if ex.name == "read_section":
            match = _HEADED_BLOCK.match(ex.output)
            if not match:
                continue
            paragraphs = match.group(3).split("\n\n")
            if paragraphs and re.match(r"(Continues on page|End of section)", paragraphs[-1]):
                paragraphs = paragraphs[:-1]
            title = re.sub(r" \(page \d+ of \d+\)$", "", match.group(2))
            passages.append(_Passage(match.group(1), title, "\n\n".join(paragraphs)))
        elif ex.name == "search_paper":
            for block in ex.output.split("\n\n")[1:]:
                hit = _SEARCH_HIT.match(block)
                if hit:
                    passages.append(_Passage(hit.group(1), "", hit.group(2)))
    return passages


def _blocks(text: str) -> list[_Passage]:
    """Split ``[id] Title\\n\\ntext`` blocks (the fulltext body and mapreduce parts)."""
    passages = []
    for block in _BLOCK_START.split(text.strip()):
        match = _HEADED_BLOCK.match(block)
        if match:
            passages.append(_Passage(match.group(1), match.group(2), match.group(3)))
    return passages


# -- text heuristics ---------------------------------------------------------


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _overlap(claim: str, sentence: str) -> float:
    claim_words = _words(claim)
    return len(claim_words & _words(sentence)) / len(claim_words) if claim_words else 0.0


def _sentences(passages: Iterable[_Passage]) -> list[tuple[str, str]]:
    return [(p.section_id, s) for p in passages for s in split_sentences(p.text)]


def _best_match(claim: str, sentences: list[tuple[str, str]]) -> tuple[str, str, float]:
    best = ("", "", 0.0)
    for section_id, sentence in sentences:
        score = _overlap(claim, sentence)
        if score > best[2]:
            best = (section_id, sentence, score)
    return best


def _truncate_words(text: str, limit: int) -> str:
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit]).rstrip(",;:") + "."


def _pick(passages: list[_Passage], pattern: re.Pattern[str]) -> _Passage | None:
    return next((p for p in passages if pattern.search(p.title)), None)


def _draft(passages: list[_Passage], *, misattribute: bool = False) -> dict[str, Any]:
    """A summary built from the sentences of what was read."""
    passages = [p for p in passages if p.text.strip()]
    if not passages:
        passages = [_Passage("", "", "The paper has no readable text.")]
    intro = _pick(passages, _INTRO) or passages[0]
    method = _pick(passages, _METHOD) or passages[min(1, len(passages) - 1)]
    results = [p for p in passages if _RESULTS.search(p.title)] or passages
    intro_sentences = split_sentences(intro.text) or [intro.text]
    method_sentences = split_sentences(method.text) or [method.text]

    candidates = [(p.section_id, s) for p in results for s in split_sentences(p.text)]
    numeric = [c for c in candidates if numbers_in(c[1])]
    others = [c for c in _sentences(passages) if c not in numeric]
    chosen = (numeric + others)[:4]
    findings = [{"claim": s, "section_ids": [sid] if sid else []} for sid, s in chosen]
    while len(findings) < 2:
        findings.append({"claim": intro_sentences[0], "section_ids": [intro.section_id]})
    if misattribute:
        _misattribute(findings[-1], [s for _, s in _sentences(passages)])

    text = " ".join(p.text for p in passages)
    words = Counter(w for w in re.findall(r"[a-z]{5,}", text.lower()) if w not in _STOPWORDS)
    acronyms = Counter(_ACRONYM.findall(text))
    contributions = [s for s in intro_sentences if _CONTRIBUTION.search(s)][:2]
    return {
        "tldr": _truncate_words(intro_sentences[0], MAX_TLDR_WORDS),
        "problem": " ".join(intro_sentences[:2]),
        "approach": " ".join(method_sentences[:2]),
        "findings": findings,
        "contributions": contributions or method_sentences[:1],
        "limitations": [s for _, s in _sentences(passages) if _LIMITATION.search(s)][:2],
        "methods": [a for a, _ in acronyms.most_common(3)],
        "datasets": [],
        "keywords": [w for w, _ in words.most_common(5)] or ["research", "paper", "summary"],
    }


def _misattribute(finding: dict[str, Any], sentences: list[str]) -> None:
    """Swap the claim's first number for another number from the paper."""
    claim = finding["claim"]
    own = numbers_in(claim)
    if not own:
        return
    elsewhere = [n for s in sentences for n in numbers_in(s) if n not in own]
    if elsewhere:
        finding["claim"] = re.sub(
            rf"(?<![\w.]){re.escape(own[0])}(?![\d])", elsewhere[0], claim, count=1
        )


def _misattributes(title: str) -> bool:
    return int(sha256_hex(title)[:8], 16) % 4 == 0


# -- responses ---------------------------------------------------------------


def _message(params: dict[str, Any], content: list[dict[str, Any]], stop_reason: str) -> Message:
    key = sha256_hex(canonical_json(params))
    for i, block in enumerate(content):
        if block["type"] == "tool_use":
            block["id"] = f"toolu_offline_{key[:16]}_{i}"
    return Message.model_validate(
        {
            "id": f"msg_offline_{key[:24]}",
            "type": "message",
            "role": "assistant",
            "model": params["model"],
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            # Zero usage: FakeAnthropic fills it in from the prompt, modelling
            # the prompt cache.
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    )


def _thinking() -> dict[str, Any]:
    return {"type": "thinking", "thinking": "", "signature": "offline"}


def _call(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": "", "name": name, "input": tool_input}


def _tool_turn(params: dict[str, Any], calls: list[dict[str, Any]]) -> Message:
    return _message(params, [_thinking(), *calls], "tool_use")


def _text_turn(params: dict[str, Any], text: str) -> Message:
    return _message(params, [_thinking(), {"type": "text", "text": text}], "end_turn")


# -- roles -------------------------------------------------------------------


def reader(params: dict[str, Any]) -> Message | None:
    """Reads the introduction, method and results, then submits a summary."""
    if not _role_is(params, "reader"):
        return None
    messages = params["messages"]
    exchanges = _exchanges(messages)
    brief = _brief(params)
    if not exchanges and not _must_finish(messages):
        outline = _OUTLINE_LINE.findall(_tagged(brief, "outline"))
        readable = [(sid, title) for sid, title, size in outline if size != "heading only"]
        picks: list[str] = []
        for pattern in (_INTRO, _METHOD, _RESULTS):
            sid = next((s for s, t in readable if pattern.search(t) and s not in picks), None)
            if sid:
                picks.append(sid)
        picks += [s for s, _ in readable if s not in picks][: _SECTIONS_TO_READ - len(picks)]
        return _tool_turn(
            params, [_call("read_section", {"section_id": s, "page": 1}) for s in picks]
        )
    title = re.search(r"^Title: (.*)$", brief, re.MULTILINE)
    misattribute = title is not None and _misattributes(title.group(1))
    draft = _draft(_read_passages(exchanges), misattribute=misattribute)
    return _tool_turn(params, [_call(SUBMIT_SUMMARY_NAME, draft)])


def critic(params: dict[str, Any]) -> Message | None:
    """Reads the cited sections, then quotes the best-matching sentence per claim."""
    if not _role_is(params, "critic"):
        return None
    messages = params["messages"]
    exchanges = _exchanges(messages)
    claims = _CLAIM_LINE.findall(_tagged(_brief(params), "claims"))
    if not exchanges and not _must_finish(messages):
        cited = dict.fromkeys(
            s.strip() for _, _, ids in claims for s in ids.split(",") if s.strip() != "none" and ids
        )
        calls = [_call("read_section", {"section_id": s, "page": 1}) for s in cited]
        calls.append(_call("search_paper", {"query": " ".join(claims[0][1].split()[:10])}))
        return _tool_turn(params, calls)
    sentences = _sentences(_read_passages(exchanges))
    checks = [_check(target, claim, sentences) for target, claim, _ in claims]
    return _tool_turn(params, [_call(SUBMIT_VERIFICATION_NAME, {"checks": checks})])


def _check(target: str, claim: str, sentences: list[tuple[str, str]]) -> dict[str, Any]:
    section_id, sentence, score = _best_match(claim, sentences)
    missing = [n for n in numbers_in(claim) if n not in numbers_in(sentence)]
    if score < 0.3:
        verdict, note = "unsupported", "No passage in the cited sections states this."
    elif missing and numbers_in(sentence):
        verdict, note = "contradicted", f"The paper gives a different figure than {missing[0]}."
    elif missing or score < 0.6:
        verdict, note = "partial", "Only part of the claim is stated in the paper."
    else:
        verdict, note = "supported", "The cited section states this."
    quote = sentence if verdict != "unsupported" else ""
    return {
        "target": target,
        "verdict": verdict,
        "evidence": quote,
        "section_id": section_id if quote else None,
        "note": note,
    }


def reviser(params: dict[str, Any]) -> Message | None:
    """Re-reads the failed claims' sections and restates them from the text."""
    if not _role_is(params, "reviser"):
        return None
    messages = params["messages"]
    brief = _brief(params)
    draft = json.loads(_tagged(brief, "summary"))
    failed = {
        t
        for t, verdict in _CHECK_LINE.findall(_tagged(brief, "failed_checks"))
        if verdict != "supported"
    }
    indexes = [int(t.split(":")[1]) for t in failed if t.startswith("finding:")]
    exchanges = _exchanges(messages)
    if not exchanges and not _must_finish(messages):
        cited = dict.fromkeys(s for i in indexes for s in draft["findings"][i]["section_ids"])
        if cited:
            return _tool_turn(
                params, [_call("read_section", {"section_id": s, "page": 1}) for s in cited]
            )
    passages = _read_passages(exchanges)
    sentences = _sentences(passages)
    for i in indexes:
        finding = draft["findings"][i]
        section_id, sentence, score = _best_match(finding["claim"], sentences)
        if score >= 0.3:
            draft["findings"][i] = {"claim": sentence, "section_ids": [section_id]}
    if len(draft["findings"]) > 2:
        draft["findings"] = [f for f in draft["findings"] if f["claim"]]
    if "tldr" in failed and sentences:
        draft["tldr"] = _truncate_words(_best_match(draft["tldr"], sentences)[1], MAX_TLDR_WORDS)
    return _tool_turn(params, [_call(SUBMIT_SUMMARY_NAME, draft)])


def fulltext(params: dict[str, Any]) -> Message | None:
    """Summarizes the whole body as JSON."""
    if not _role_is(params, "fulltext"):
        return None
    passages = _blocks(_tagged(_brief(params), "body"))
    return _text_turn(params, canonical_json(_draft(passages)))


def mapreduce_notes(params: dict[str, Any]) -> Message | None:
    """Notes every sentence with a number, and each section's first sentence."""
    if not _role_is(params, "mapreduce_notes"):
        return None
    lines = []
    for passage in _blocks(_tagged(_brief(params), "part")):
        for i, sentence in enumerate(split_sentences(passage.text)):
            if i == 0 or numbers_in(sentence):
                lines.append(f"[{passage.section_id}] {sentence}")
    return _text_turn(params, "\n".join(lines) or "Nothing of substance in this part.")


def mapreduce_combine(params: dict[str, Any]) -> Message | None:
    """Builds the summary from the notes' sentences."""
    if not _role_is(params, "mapreduce_combine"):
        return None
    brief = _brief(params)
    titles = {sid: title for sid, title, _ in _OUTLINE_LINE.findall(_tagged(brief, "outline"))}
    grouped: dict[str, list[str]] = {}
    for section_id, sentence in _NOTE_LINE.findall(_tagged(brief, "notes")):
        grouped.setdefault(section_id, []).append(sentence)
    passages = [_Passage(sid, titles.get(sid, ""), " ".join(s)) for sid, s in grouped.items()]
    return _text_turn(params, canonical_json(_draft(passages)))


RESPONDERS: tuple[OfflineResponder, ...] = (
    reader,
    critic,
    reviser,
    fulltext,
    mapreduce_notes,
    mapreduce_combine,
)
"""Every summarization role; ``precis.offline`` wraps them for ``FakeAnthropic``."""
