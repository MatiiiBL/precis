"""Passage search over one paper, for the agents' ``search_paper`` tool.

The document is cut into passages of about ``PASSAGE_WORDS`` words, at
sentence boundaries and never across sections, so every hit can be cited by
section and read in context with ``read_section``. Passages are ranked with
Okapi BM25: a term counts for more the rarer it is in this paper (idf), with
diminishing returns for repeats (``k1``) and a correction for passage length
(``b``). A paper has a few hundred passages, so the index is built in a few
milliseconds and kept in memory for the length of an agent run.

Figure and table captions are not indexed: they have no section, and the
agents read them with ``read_captions``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, replace

from precis.models import Document

PASSAGE_WORDS = 120
K1 = 1.5
B = 0.75

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$(\[])")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        *("a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have"),
        *("in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was"),
        *("were", "which", "with", "we", "our", "these", "those", "than", "can", "not"),
    }
)


@dataclass(frozen=True, slots=True)
class Passage:
    """A searchable span of one section; ``score`` is set on search results."""

    id: str  # "<section_id>#p<n>", e.g. "s2.1#p3"
    section_id: str
    text: str
    score: float = 0.0


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs, minus stopwords, with plurals folded.

    "KV-caches" becomes ["kv", "cache"], so a query matches the singular,
    the plural and the hyphenated compound alike.
    """
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _stem(token: str) -> str:
    # Deliberately crude: query and passages go through the same folding, so
    # "analysis" -> "analysi" is harmless, and "GPUs" -> "gpu" is what matters.
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


class PaperIndex:
    """BM25 over the passages of one ``Document``."""

    def __init__(self, doc: Document, *, passage_words: int = PASSAGE_WORDS) -> None:
        self.passages = [
            Passage(id=f"{section.id}#p{n}", section_id=section.id, text=text)
            for section in doc.sections
            for n, text in enumerate(split_passages(section.text, passage_words), 1)
        ]
        self._tf = [Counter(tokenize(p.text)) for p in self.passages]
        self._lengths = [sum(tf.values()) for tf in self._tf]
        self._avg_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.passages)
        self._idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[Passage]:
        """The ``k`` best passages for ``query``, best first; only passages that match."""
        terms = [t for t in set(tokenize(query)) if t in self._idf]
        if k <= 0 or not terms:
            return []
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            norm = K1 * (1 - B + B * self._lengths[i] / self._avg_length)
            score = sum(self._idf[t] * tf[t] * (K1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [replace(self.passages[i], score=round(score, 4)) for score, i in scored[:k]]


def split_passages(text: str, words: int = PASSAGE_WORDS) -> list[str]:
    """Cut one section's text into passages of about ``words`` words.

    Passages end at sentence boundaries; a sentence longer than half again
    the target is cut by word count, and a short tail joins the passage
    before it rather than standing alone.
    """
    sentences = [
        piece
        for paragraph in text.split("\n\n")
        for sentence in _SENTENCE_END.split(paragraph)
        for piece in _cut_long(sentence.split(), words)
    ]
    passages: list[list[str]] = []
    current: list[str] = []
    for sentence in sentences:
        size = len(sentence.split())
        if current and len(current) + size > words * 1.5 and len(current) >= words // 2:
            passages.append(current)
            current = []
        current.extend(sentence.split())
        if len(current) >= words:
            passages.append(current)
            current = []
    if current:
        if passages and len(current) < words // 3:
            passages[-1].extend(current)
        else:
            passages.append(current)
    return [" ".join(p) for p in passages]


def _cut_long(tokens: list[str], words: int) -> list[str]:
    """A sentence as one piece, or in ``words``-sized pieces when it is far too long."""
    if len(tokens) <= words * 1.5:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + words]) for i in range(0, len(tokens), words)]
