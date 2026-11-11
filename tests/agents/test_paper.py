from precis.agents.paper import (
    PaperView,
    canonical_number,
    normalize_text,
    numbers_in,
    split_pages,
    split_sentences,
)
from precis.util import estimate_tokens

from .helpers import DESIGN_1, DESIGN_2, RESULTS, make_doc, make_meta


def view(page_tokens: int = 40) -> PaperView:
    return PaperView(make_meta(), make_doc(), page_tokens=page_tokens)


def test_pages_break_between_paragraphs() -> None:
    pages = split_pages(f"{DESIGN_1}\n\n{DESIGN_2}", 40)
    assert pages == [DESIGN_1, DESIGN_2]
    assert split_pages(f"{DESIGN_1}\n\n{DESIGN_2}", 1000) == [f"{DESIGN_1}\n\n{DESIGN_2}"]


def test_long_paragraphs_break_between_sentences_then_anywhere() -> None:
    pages = split_pages(RESULTS, 20)
    assert len(pages) == 3
    assert all(estimate_tokens(p) <= 20 for p in pages)
    giant = "x" * 1000
    assert [len(p) for p in split_pages(giant, 100)] == [400, 400, 200]
    assert split_pages("", 100) == []
    assert split_pages("\n\n  \n\n", 100) == []


def test_sentences() -> None:
    assert split_sentences(RESULTS) == [
        "ReuseNet reduces LLC misses by 18.4% on average over LRU.",
        "It improves IPC by 6.2% over LRU and by 1.9% over Hawkeye.",
        "The largest gain, 41.3% fewer misses, is on mcf.",
    ]


def test_numbers_are_canonical_and_small_integers_skipped() -> None:
    assert canonical_number("1,024") == "1024"
    assert canonical_number("1.50") == "1.5"
    assert canonical_number("07") == "7"
    assert canonical_number("2.0") == "2"
    assert numbers_in("3 caches, 16 ways, 18.4% fewer misses, 2,048 entries, x86 and L2") == [
        "16",
        "18.4",
        "2048",
    ]


def test_quotes_match_despite_formatting_but_not_paraphrase() -> None:
    v = view()
    assert v.contains_quote("ReuseNet reduces LLC misses by 18.4% on average over LRU.")
    assert v.contains_quote('"reuseNet  reduces LLC\nmisses by 18.4%"')
    assert v.contains_quote("ReuseNet reduces LLC misses ... over LRU")
    assert v.contains_quote(f"Hawkeye learns from Belady{chr(0x2019)}s optimal policy")
    assert v.contains_quote("Storage: ReuseNet 12.5 KB")  # captions count
    assert not v.contains_quote("ReuseNet reduces LLC misses by 19.4% on average over LRU.")
    assert not v.contains_quote("ReuseNet cuts misses by a fifth.")
    assert not v.contains_quote("...")


def test_normalization_folds_typography() -> None:
    nbsp, left, right, en_dash = chr(0xA0), chr(0x201C), chr(0x201D), chr(0x2013)
    assert normalize_text(f"A{nbsp}B {left}C{right} {en_dash} D") == 'a b "c" - d'


def test_outline_shows_ids_sizes_and_nesting() -> None:
    outline = view().outline()
    lines = outline.splitlines()
    assert lines[0].startswith("Body: ")
    assert "s2  Background  (heading only)" in lines
    assert any(line.startswith("  s2.1  Replacement policies  (") for line in lines)
    assert any(line.startswith("s3  Design of ReuseNet") and "2 pages" in line for line in lines)
    assert lines[-1] == "Captions: 1 figures, 1 tables (read_captions)."


def test_header_withholds_the_abstract_on_request() -> None:
    v = view()
    assert "We present ReuseNet" in v.header(include_abstract=True)
    withheld = v.header(include_abstract=False)
    assert "We present ReuseNet" not in withheld
    assert "Abstract: withheld" in withheld
    assert "Categories: cs.AR (also cs.LG)" in withheld


def test_long_author_lists_are_shortened() -> None:
    meta = make_meta().model_copy(update={"authors": [f"Author {i}" for i in range(9)]})
    header = PaperView(meta, make_doc(), page_tokens=40).header(include_abstract=False)
    assert "Author 5, et al. (9 authors)" in header
    assert "Author 6" not in header


def test_subsections_and_body() -> None:
    v = view()
    assert [s.id for s in v.subsections("s4")] == ["s4.1", "s4.2"]
    assert v.subsections("s1") == []
    assert v.next_section("s5") is None
    body = v.body()
    assert body.startswith("[s1] Introduction\n\n")
    assert "[tab1] (table) Storage" in body
