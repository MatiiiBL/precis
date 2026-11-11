"""Deciding whether a parsed document is worth summarizing.

A document that is too short or mostly garbage would still "succeed" as a
``Document`` and then waste agent turns producing a confident summary of
nothing. The fetch stage fails such papers with the reason instead, so they
show up in ``precis status`` and can be inspected.
"""

from __future__ import annotations

import re
import unicodedata

from precis.models import Document
from precis.text.document import ParseError

# Below this the "paper" is a stub: a withdrawn notice, an extended abstract
# without a body, or a conversion that lost nearly everything.
MIN_BODY_TOKENS = 500

# Failed font decoding shows up either as characters that are not text at
# all (control codes, private-use glyph ids, replacement characters) or as
# letters that do not form English words. English prose is about one third
# function words; garbled text has almost none.
MAX_BAD_CHAR_FRACTION = 0.05
MIN_FUNCTION_WORD_FRACTION = 0.10
MAX_RUN_ON_FRACTION = 0.05  # tokens over 25 characters: lost word spacing

_FUNCTION_WORDS = frozenset(
    {
        *("the", "of", "and", "to", "in", "a", "is", "for", "that", "with", "on", "we"),
        *("are", "by", "this", "as", "be", "an", "it", "from", "at", "or", "which", "can"),
        *("our", "these", "their", "than", "its", "not", "into", "each", "when", "has"),
        *("have", "was", "were", "also", "more"),
    }
)
_WORD = re.compile(r"[^\W\d_]+")


class QualityError(ParseError):
    """The document parsed, but is too short or too garbled to summarize."""


def check_document(doc: Document) -> None:
    """Raise ``QualityError`` explaining why ``doc`` is unusable; return if it is fine."""
    if doc.n_tokens < MIN_BODY_TOKENS:
        raise QualityError(
            f"only {doc.n_tokens} tokens of body text (minimum {MIN_BODY_TOKENS}); "
            "the paper is a stub or the conversion lost its text"
        )
    reason = garbage_reason("\n".join(section.text for section in doc.sections))
    if reason is not None:
        raise QualityError(reason)


def garbage_reason(text: str) -> str | None:
    """Why ``text`` looks like failed text extraction, or ``None`` if it reads as prose."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return "no text"
    bad = sum(1 for c in chars if _is_bad_char(c))
    if bad / len(chars) > MAX_BAD_CHAR_FRACTION:
        return (
            f"{bad / len(chars):.0%} of the characters are control, private-use or "
            "replacement characters (failed font decoding)"
        )
    tokens = text.split()
    run_on = sum(1 for t in tokens if len(t) > 25)
    if tokens and run_on / len(tokens) > MAX_RUN_ON_FRACTION:
        return f"{run_on / len(tokens):.0%} of the tokens are run-on strings (lost word spacing)"
    words = _WORD.findall(text.lower())
    if words:
        function = sum(1 for w in words if w in _FUNCTION_WORDS) / len(words)
        if function < MIN_FUNCTION_WORD_FRACTION:
            return (
                f"only {function:.0%} of the words are common English function words "
                "(garbled text from failed font decoding?)"
            )
    return None


def _is_bad_char(char: str) -> bool:
    return char == "�" or unicodedata.category(char) in {"Cc", "Co", "Cn", "Cs"}
