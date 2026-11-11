"""Character-level cleanup shared by the HTML and PDF parsers."""

from __future__ import annotations

import re
import unicodedata

# Typographic ligatures that PDF fonts emit as single code points, odd spaces,
# and invisible characters that only get in the way of search and token
# counts. Keyed by code point so the table stays readable.
_TRANSLATION: dict[int, str | None] = {
    0xFB00: "ff",
    0xFB01: "fi",
    0xFB02: "fl",
    0xFB03: "ffi",
    0xFB04: "ffl",
    0xFB05: "st",
    0xFB06: "st",
    0x00A0: " ",  # no-break space
    0x2009: " ",  # thin space
    0x202F: " ",  # narrow no-break space
    0x00AD: None,  # soft hyphen
    0x200B: None,  # zero-width space (LaTeXML puts these in math)
    0x200C: None,
    0x200D: None,
    0x2060: None,
    0xFEFF: None,
}

_SPACE = re.compile(r"\s+")
# Mathematical italic/bold letters (U+1D400-U+1D7FF) that PDF math fonts map
# to; as plain letters they match a search for the same symbol in prose.
_MATH_ALPHANUMERIC = re.compile("[\U0001d400-\U0001d7ff]")
# UTF-16 surrogate code points (built with chr so the source stays valid UTF-8).
_SURROGATE = re.compile(f"[{chr(0xD800)}-{chr(0xDFFF)}]")


def clean_chars(text: str) -> str:
    """Repair surrogates, expand ligatures, plain-letter math italics, drop invisible
    characters; keep newlines."""
    text = fix_surrogates(text)
    text = _MATH_ALPHANUMERIC.sub(lambda m: unicodedata.normalize("NFKC", m[0]), text)
    return text.translate(_TRANSLATION)


def fix_surrogates(text: str) -> str:
    """Recombine UTF-16 surrogate pairs into characters and drop lone halves.

    pypdf sometimes returns a math letter such as U+1D44E as its two UTF-16
    halves, or only the high half ('\\ud835') when the low one was lost. A
    Python string may hold them, but they cannot be encoded as UTF-8, so
    one such paper made the store fail to write its document.
    """
    if not _SURROGATE.search(text):
        return text
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "ignore")


def squash(text: str) -> str:
    """Collapse every run of whitespace (including newlines) to one space."""
    return _SPACE.sub(" ", clean_chars(text)).strip()


def join_paragraphs(paragraphs: list[str]) -> str:
    """The section text format: whitespace-normalized paragraphs separated by a blank line."""
    return "\n\n".join(p for p in (squash(p) for p in paragraphs) if p)
