import random
import string

import pytest

from precis.models import Document
from precis.text.document import SectionDraft, build_document
from precis.text.quality import MIN_BODY_TOKENS, QualityError, check_document, garbage_reason

from .helpers import prose


def _doc(*paragraphs: str) -> Document:
    return build_document("p", "pdf", [SectionDraft("Body", 1, list(paragraphs))], [], "sha")


def test_a_normal_paper_passes() -> None:
    check_document(_doc(*(prose(f"cache{i}", sentences=10) for i in range(5))))


def test_a_stub_fails_with_its_size() -> None:
    expected = rf"only \d+ tokens of body text \(minimum {MIN_BODY_TOKENS}\)"
    with pytest.raises(QualityError, match=expected):
        check_document(_doc(prose("short", sentences=2)))


def test_private_use_glyphs_from_failed_font_decoding_fail() -> None:
    glyphs = "".join(chr(0xE000 + i % 200) for i in range(4000))
    with pytest.raises(QualityError, match="private-use"):
        check_document(_doc(prose("x", sentences=5), glyphs))


def test_text_with_letters_but_no_words_fails() -> None:
    # A wrong character map still produces letters, just not English.
    rng = random.Random(0)
    gibberish = " ".join(
        "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 9)))
        for _ in range(1500)
    )
    with pytest.raises(QualityError, match="function words"):
        check_document(_doc(gibberish))


def test_text_that_lost_its_spaces_fails() -> None:
    run_on = " ".join(["thecachereplacementpolicyevictsblocks"] * 400)
    assert "run-on" in (garbage_reason(run_on) or "")


def test_math_and_symbols_are_not_garbage() -> None:
    # Symbols PDFs extract from math: <=, infinity, arrow, alpha, beta, times.
    symbols = " ".join(chr(c) for c in (0x2264, 0x221E, 0x2192, 0x3B1, 0x3B2, 0xD7))
    math = f" $\\sum_{{i}}x_{{i}}^{{2}}$ {symbols} 10x 3.2% (+-0.1)"
    assert garbage_reason(prose("fp", sentences=8) + math * 20) is None
