import gzip
from collections.abc import Sequence
from datetime import date

import pytest

from precis.models import Document
from precis.text.document import ParseError
from precis.text.html import parse_html
from precis.text.pdf import parse_pdf, read_pdf
from precis.text.quality import check_document

from .helpers import FIXTURES, PdfLine, make_pdf


def prose(marker: str) -> list[str]:
    """Four lines of running text, the first starting with ``marker``."""
    return [
        f"{marker} servers spend a large share of their time waiting for memory and",
        "the last level cache is the main defence against that latency in the",
        "systems that we study here, so the policy that decides which block to",
        "evict matters for performance and for the energy that we spend on it.",
    ]


def paginate(lines: Sequence[PdfLine], per_page: int = 40) -> list[list[PdfLine]]:
    """Pages with a running header and a page number, like a conference paper."""
    pages = [list(lines[i : i + per_page]) for i in range(0, len(lines), per_page)]
    return [["MICRO '25, October 2025, Seoul, Korea", *p, str(n)] for n, p in enumerate(pages, 1)]


def parse(lines: Sequence[PdfLine], **info: str) -> Document:
    return parse_pdf(make_pdf(paginate(lines), info=info or None), "2501.00001")


def text_of(doc: Document) -> str:
    return "\n".join([s.text for s in doc.sections] + [c.text for c in doc.captions])


ACM_PAPER: list[PdfLine] = [
    ("Learned Cache Replacement for Servers", 18.0),
    "Ada Author and Bob Builder",
    "ABSTRACT",
    "ABSTRACTMARK we study replacement policies for very large shared caches.",
    "1 INTRODUCTION",
    *prose("INTROMARK"),
    "2 BACKGROUND",
    *prose("BACKMARK"),
    "2.1 Cache Organization",
    *prose("ORGMARK"),
    "3 DESIGN",
    *prose("DESIGNMARK"),
    "Figure 1: The replacement pipeline.",
    "L1",
    "L2 Cache",
    "DRAM",
    "Core 0",
    "The pipeline in Figure 1 has three stages and each of them is simple to",
    "build in hardware, which keeps the area overhead of the design small.",
    "4 EVALUATION",
    *prose("EVALMARK"),
    "ACKNOWLEDGMENTS",
    "THANKSMARK we thank the funding agency for its generous support of this work.",
    "REFERENCES",
    "[1] A. Smith. REFMARK Caches and how to replace their blocks. 2020.",
    "[2] B. Jones. Another paper about memory systems. 2021.",
]


def test_numbered_headings_become_sections() -> None:
    doc = parse(ACM_PAPER)
    assert [(s.id, s.level, s.title) for s in doc.sections] == [
        ("s1", 1, "INTRODUCTION"),
        ("s2", 1, "BACKGROUND"),
        ("s2.1", 2, "Cache Organization"),
        ("s3", 1, "DESIGN"),
        ("s4", 1, "EVALUATION"),
    ]
    assert doc.sections[0].text.startswith("INTROMARK servers spend")
    assert doc.source_format == "pdf"


def test_front_matter_acknowledgements_references_and_furniture_are_removed() -> None:
    text = text_of(parse(ACM_PAPER))
    for marker in ("ABSTRACTMARK", "Ada Author", "THANKSMARK", "REFMARK", "MICRO '25"):
        assert marker not in text
    assert "\n\n2\n\n" not in f"\n\n{text}\n\n"


def test_captions_are_moved_out_and_figure_labels_dropped() -> None:
    doc = parse(ACM_PAPER)
    assert [(c.id, c.text) for c in doc.captions] == [
        ("fig1", "Figure 1: The replacement pipeline.")
    ]
    design = doc.sections[3].text
    assert "DRAM" not in design
    assert "Core 0" not in design
    assert "The pipeline in Figure 1 has three stages" in design


def test_hyphenation_across_lines_uses_the_documents_vocabulary() -> None:
    lines: list[PdfLine] = [
        "1 INTRODUCTION",
        "We describe the column-major layout of the cache and how the archi-",
        "tecture handles it, then show why a column-",
        "major order helps the CKKS-",
        "friendly schedule that the rest of this paper builds on in detail.",
        *prose("MORE"),
        "2 DESIGN",
        *prose("DESIGN"),
    ]
    text = parse(lines).sections[0].text
    assert "the architecture handles it" in text
    assert "a column-major order" in text
    assert "the CKKS-friendly schedule" in text


def test_numbered_list_items_and_axis_ticks_do_not_become_headings() -> None:
    lines: list[PdfLine] = [
        "1 Introduction",
        *prose("INTRO"),
        "1. We propose a policy that learns which blocks to keep in the cache.",
        "2. We build it in hardware and show that the overhead is very small.",
        "3. We evaluate it on servers and find that it beats the baseline.",
        *prose("AFTER"),
        "2 Design",
        *prose("DESIGN"),
        "2",
        "Hit Rate",
        "4",
        "6",
        *prose("PLOT"),
        "3 Evaluation",
        *prose("EVAL"),
    ]
    doc = parse(lines)
    assert [s.title for s in doc.sections] == ["Introduction", "Design", "Evaluation"]
    assert "We build it in hardware" in doc.sections[0].text


def test_ieee_roman_headings_with_small_caps_letters_and_run_ins() -> None:
    lines: list[PdfLine] = [
        "I. I NTRODUCTION",
        *prose("INTRO"),
        "II. R ELATED W ORK",
        "A. Learned Policies",
        *prose("LEARNED"),
        "B. Prefetching",
        "1) Stride Prefetchers: Stride prefetchers detect constant strides in",
        *prose("STRIDE"),
        "2) Correlation Prefetchers: These remember pairs of addresses seen",
        *prose("CORRELATION"),
        "III. D ESIGN",
        *prose("DESIGN"),
    ]
    doc = parse(lines)
    assert [(s.id, s.title) for s in doc.sections] == [
        ("s1", "INTRODUCTION"),
        ("s2", "RELATED WORK"),
        ("s2.1", "Learned Policies"),
        ("s2.2", "Prefetching"),
        ("s2.2.1", "Stride Prefetchers"),
        ("s2.2.2", "Correlation Prefetchers"),
        ("s3", "DESIGN"),
    ]
    stride = doc.sections[4].text
    assert stride.startswith("Stride prefetchers detect constant strides in STRIDE servers")


def test_wrapped_split_and_run_in_numbered_headings() -> None:
    lines: list[PdfLine] = [
        "1 Introduction",
        *prose("INTRO"),
        "2 SPEC CPU26 vs. SPEC CPU17, DCPerf, MLPerf,",
        "Agentic AI, GEMM",
        *prose("COMPARE"),
        "2.1 Matrix-Vector Mul-",
        "tiplication",
        *prose("MATVEC"),
        "2.1.1 Memory Requirements. Working set size is an important attribute of the",
        *prose("WSS"),
        "3",
        "Results",
        *prose("RESULTS"),
    ]
    doc = parse(lines)
    assert [(s.id, s.title) for s in doc.sections] == [
        ("s1", "Introduction"),
        ("s2", "SPEC CPU26 vs. SPEC CPU17, DCPerf, MLPerf, Agentic AI, GEMM"),
        ("s2.1", "Matrix-Vector Multiplication"),
        ("s2.1.1", "Memory Requirements"),
        ("s3", "Results"),
    ]
    assert doc.sections[3].text.startswith("Working set size is an important attribute")


def test_unnumbered_all_caps_headings_are_the_fallback() -> None:
    lines: list[PdfLine] = [
        ("A Paper Without Numbers", 16.0),
        "INTRODUCTION",
        *prose("INTRO"),
        "OUR METHOD",
        *prose("METHOD"),
        "RESULTS",
        *prose("RESULTS"),
    ]
    doc = parse(lines)
    assert [s.title for s in doc.sections] == ["INTRODUCTION", "OUR METHOD", "RESULTS"]


def test_text_without_headings_is_split_into_parts_without_the_abstract() -> None:
    # Every line differs: identical lines at the top of several pages would
    # (rightly) look like a running header.
    tags = [f"q{chr(97 + n // 26)}{chr(97 + n % 26)}" for n in range(60)]
    body = [
        line
        for tag in tags
        for line in (
            f"{tag} servers spend a large share of their time waiting for memory and",
            f"the last level cache of {tag} is the main defence against that latency",
            f"in the systems that we study, so the {tag} policy that decides which",
            f"block to evict matters for performance and for the {tag} energy budget.",
        )
    ]
    lines: list[PdfLine] = [
        "Abstract",
        "ABSTRACTMARK is a summary of the paper and of what it finds, and it",
        "ends here.",
        *body,
    ]
    doc = parse(lines)
    assert len(doc.sections) >= 3
    assert [s.title for s in doc.sections] == [f"Part {i}" for i in range(1, len(doc.sections) + 1)]
    assert all(s.n_tokens <= 1600 for s in doc.sections)
    text = text_of(doc)
    assert "ABSTRACTMARK" not in text
    assert all(f"{tag} servers spend" in text for tag in tags)


def test_lettered_appendix_after_references_is_kept() -> None:
    lines: list[PdfLine] = [
        "1 Introduction",
        *prose("INTRO"),
        "2 Conclusion",
        *prose("CONCLUSION"),
        "References",
        "[1] A. Smith. REFMARK Caches and how to replace their blocks. 2020.",
        "[2] B. Jones. Another paper about memory systems. 2021.",
        "A Proof of the Bound",
        *prose("PROOF"),
        "A.1 A Lemma",
        *prose("LEMMA"),
        "B More Results",
        *prose("MORE"),
    ]
    doc = parse(lines)
    assert [(s.id, s.title) for s in doc.sections] == [
        ("s1", "Introduction"),
        ("s2", "Conclusion"),
        ("a1", "Proof of the Bound"),
        ("a1.1", "A Lemma"),
        ("a2", "More Results"),
    ]
    assert "REFMARK" not in text_of(doc)


def test_appendix_announced_by_name_before_the_references() -> None:
    lines: list[PdfLine] = [
        "1 Introduction",
        *prose("INTRO"),
        "2 Conclusion",
        *prose("CONCLUSION"),
        "APPENDIX",
        *prose("APPENDIXMARK"),
        "REFERENCES",
        "[1] A. Smith. REFMARK Caches. 2020.",
    ]
    doc = parse(lines)
    assert [(s.id, s.title) for s in doc.sections][-1] == ("a1", "APPENDIX")
    assert doc.sections[-1].text.startswith("APPENDIXMARK")
    assert "APPENDIXMARK" not in doc.sections[1].text


def test_unusable_files_raise_parse_errors() -> None:
    with pytest.raises(ParseError, match="not a PDF"):
        parse_pdf(b"<html>not a pdf</html>", "x")
    with pytest.raises(ParseError, match="unreadable PDF"):
        parse_pdf(b"%PDF-1.4\n%garbage without objects", "x")
    with pytest.raises(ParseError, match="no extractable text"):
        parse_pdf(make_pdf([[]]), "x")


def test_front_matter_title_abstract_and_date() -> None:
    info = {"Title": "Learned Cache Replacement", "CreationDate": "D:20250704120000Z"}
    doc, front = read_pdf(make_pdf(paginate(ACM_PAPER), info=info), "x")
    assert doc.sections[0].title == "INTRODUCTION"
    assert front.title == "Learned Cache Replacement"
    assert front.abstract == (
        "ABSTRACTMARK we study replacement policies for very large shared caches."
    )
    assert front.created == date(2025, 7, 4)
    assert front.license is None
    # Without usable metadata the largest text on the first page is the title.
    for bad in (None, {"Title": "paper.dvi"}, {"CreationDate": "garbage"}):
        _, front = read_pdf(make_pdf(paginate(ACM_PAPER), info=bad), "x")
        assert front.title == "Learned Cache Replacement for Servers"
        assert front.created is None


def test_real_ieee_paper_matches_its_html_outline() -> None:
    """Two-column IEEE layout with small-caps headings: same outline as the HTML."""
    pdf = parse_pdf((FIXTURES / "2508.21267v1.pdf").read_bytes(), "2508.21267")
    html = parse_html(gzip.decompress((FIXTURES / "2508.21267v1.html.gz").read_bytes()), "x")
    check_document(pdf)
    assert [s.id for s in pdf.sections] == [s.id for s in html.sections]
    assert [s.title.lower() for s in pdf.sections] == [s.title.lower() for s in html.sections]
    assert 0.9 < pdf.n_tokens / html.n_tokens < 1.5
    text = text_of(pdf)
    assert "Neuromorphic computing, particularly represented" in pdf.sections[0].text
    assert "EFERENCES" not in text  # the heading, in small caps or not
    assert len(pdf.captions) >= 8


def test_real_preprint_with_lettered_appendix_after_author_year_references() -> None:
    doc = parse_pdf((FIXTURES / "2610.02233v2-p1-13.pdf").read_bytes(), "2610.02233")
    check_document(doc)
    ids = [s.id for s in doc.sections]
    assert ids[:5] == ["s1", "s2", "s3", "s4", "s4.1"]
    assert ids[-4:] == ["a1", "a1.1", "a1.2", "a2"]
    text = text_of(doc)
    assert "Zefan Cai, Yichi Zhang" not in text  # the first reference
    assert "Cross-context KV-cache reuse predicts" not in text  # the abstract
    _, front = read_pdf((FIXTURES / "2610.02233v2-p1-13.pdf").read_bytes(), "x")
    assert front.title == "Budgeted Cache Repair for Cross-Context KV-Cache Reuse"
    assert front.license == "http://creativecommons.org/licenses/by/4.0/"
    assert front.abstract.startswith("Cross-context KV-cache reuse predicts")
