import gzip

import pytest

from precis.models import Document
from precis.text.document import ParseError
from precis.text.html import html_license, html_unusable_reason, parse_html
from precis.text.quality import check_document

from .helpers import FIXTURES, latexml_page, para, prose, section


def _parse(body: str) -> Document:
    return parse_html(latexml_page(body), "2501.00001")


def _all_text(doc: Document) -> str:
    return "\n".join([s.text for s in doc.sections] + [c.text for c in doc.captions])


def test_sections_subsections_and_appendices_get_stable_ids() -> None:
    doc = _parse(
        section(
            "S1",
            "1",
            "Introduction",
            para("Intro text."),
            section("S1.SS1", "1.1", "Scope", para("Scope text."), kind="subsection", heading="h3"),
            section(
                "S1.SS2",
                "1.2",
                "Plan",
                para("Plan text."),
                section(
                    "S1.SS2.SSS1",
                    "1.2.1",
                    "Detail",
                    para("Detail text."),
                    kind="subsubsection",
                    heading="h4",
                ),
                kind="subsection",
                heading="h3",
            ),
        )
        + section("S2", "2", "Design", para("Design text."))
        + section("A1", "Appendix A", "Proofs", para("Proof text."), kind="appendix")
    )
    assert [(s.id, s.level, s.title) for s in doc.sections] == [
        ("s1", 1, "Introduction"),
        ("s1.1", 2, "Scope"),
        ("s1.2", 2, "Plan"),
        ("s1.2.1", 3, "Detail"),
        ("s2", 1, "Design"),
        ("a1", 1, "Proofs"),
    ]
    # A section's text is its own, not its subsections'.
    assert doc.sections[0].text == "Intro text."
    assert doc.source_format == "html"
    assert doc.n_tokens == sum(s.n_tokens for s in doc.sections)


def test_front_matter_acknowledgements_and_bibliography_are_dropped() -> None:
    doc = _parse(
        section("S1", "1", "Introduction", para("Body text."))
        + section("Sx1", "", "Acknowledgments", para("ACK-TEXT thanks the funders."))
        + '<div class="ltx_acknowledgements">ACKS-DIV-TEXT</div>'
        + '<section class="ltx_bibliography"><h2 class="ltx_title">References</h2>'
        '<ul class="ltx_biblist"><li class="ltx_bibitem">BIB-TEXT 2020.</li></ul></section>'
    )
    text = _all_text(doc)
    assert [s.title for s in doc.sections] == ["Introduction"]
    for marker in ("ABSTRACT-TEXT", "KEYWORD-TEXT", "Ada Author", "A Paper Title", "NAV-TEXT"):
        assert marker not in text
    for marker in ("ACK-TEXT", "ACKS-DIV-TEXT", "BIB-TEXT"):
        assert marker not in text


def test_math_is_kept_as_latex_and_equations_keep_their_number() -> None:
    inline = '<math alttext="O(n^{2})" display="inline"><mi>O</mi></math>'
    equation = (
        '<table class="ltx_equation ltx_eqn_table"><tr class="ltx_equation ltx_eqn_row">'
        '<td class="ltx_eqn_cell"><math alttext="\\displaystyle E=mc^{2}" display="block">'
        "<mi>E</mi></math></td>"
        '<td class="ltx_eqn_cell"><span class="ltx_tag ltx_tag_equation">(3)</span></td>'
        "</tr></table>"
    )
    body = (
        f'<div class="ltx_para"><p class="ltx_p">The cost is {inline} per step, so</p>'
        f'{equation}<p class="ltx_p">holds for every block.</p></div>'
    )
    doc = _parse(section("S1", "1", "Model", body))
    assert doc.sections[0].text == (
        "The cost is $O(n^{2})$ per step, so $E=mc^{2}$ (3) holds for every block."
    )


def test_footnotes_stay_inline_without_their_marks() -> None:
    note = (
        '<span class="ltx_note ltx_role_footnote"><sup class="ltx_note_mark">1</sup>'
        '<span class="ltx_note_outer"><span class="ltx_note_content">'
        '<sup class="ltx_note_mark">1</sup><span class="ltx_tag ltx_tag_note">1</span>'
        "Measured at 2 GHz.</span></span></span>"
    )
    doc = _parse(section("S1", "1", "Setup", para(f"We run the simulator{note} on all traces.")))
    assert doc.sections[0].text == (
        "We run the simulator (Footnote: Measured at 2 GHz.) on all traces."
    )


def test_figures_and_tables_become_captions_outside_the_text() -> None:
    figure = (
        '<figure class="ltx_figure"><img src="x.png">'
        '<figure class="ltx_figure ltx_figure_panel"><figcaption class="ltx_caption">'
        '<span class="ltx_tag">(a)</span> Left panel</figcaption></figure>'
        '<figcaption class="ltx_caption"><span class="ltx_tag">Figure 1: </span>'
        "Speedup over the baseline.</figcaption></figure>"
    )
    table = (
        '<figure class="ltx_table"><figcaption class="ltx_caption">'
        '<span class="ltx_tag">Table 1: </span>Configuration.</figcaption>'
        '<table class="ltx_tabular"><tr><th>Parameter</th><th>Value</th></tr>'
        "<tr><td>L1 size</td><td>32 KB</td></tr><tr><td></td><td></td></tr></table></figure>"
    )
    doc = _parse(section("S1", "1", "Results", para("See below."), figure, table))
    assert [(c.id, c.kind) for c in doc.captions] == [("fig1", "figure"), ("tab1", "table")]
    assert doc.captions[0].text == "Figure 1: Speedup over the baseline. (a) Left panel"
    assert doc.captions[1].text == "Table 1: Configuration.\nParameter | Value\nL1 size | 32 KB"
    assert doc.sections[0].text == "See below."


def test_paragraphs_lists_run_in_titles_and_unconverted_macros() -> None:
    items = (
        '<ul class="ltx_itemize">'
        '<li class="ltx_item"><span class="ltx_tag ltx_tag_item">•</span>'
        '<div class="ltx_para"><p class="ltx_p">First   contribution.</p></div></li>'
        '<li class="ltx_item"><span class="ltx_tag ltx_tag_item">•</span>'
        '<div class="ltx_para"><p class="ltx_p">Second contribution.</p></div></li></ul>'
    )
    run_in = (
        '<section class="ltx_paragraph"><h4 class="ltx_title ltx_title_paragraph">Motivation.</h4>'
        + para('Caches <span class="ltx_ERROR undefined">\\cellcolor</span>miss.')
        + "</section>"
    )
    contributions = f'<div class="ltx_para"><p class="ltx_p">We contribute:</p>{items}</div>'
    doc = _parse(section("S1", "1", "Introduction", para("One\n  two."), contributions, run_in))
    assert doc.sections[0].text.split("\n\n") == [
        "One two.",
        "We contribute:",
        "• First contribution.",
        "• Second contribution.",
        "Motivation. Caches miss.",
    ]


def test_a_run_in_acknowledgment_paragraph_is_dropped() -> None:
    """2507.10912: a bold "Acknowledgment" lead-in, not a section, in the conclusion."""
    acknowledgment = (
        '<div class="ltx_para"><p class="ltx_p">'
        '<span class="ltx_text ltx_font_bold">Acknowledgment</span> This work is sponsored '
        "by the National Science Foundation under Grant No. CCF2403134.</p></div>"
    )
    doc = _parse(
        section("S6", "6", "Conclusions", para("FuseMap improves ADP by 9%."), acknowledgment)
    )
    assert doc.sections[0].text == "FuseMap improves ADP by 9%."


def test_front_matter_notes_are_not_mistaken_for_body_text() -> None:
    note = (
        '<span class="ltx_note ltx_role_footnotetext"><span class="ltx_note_outer">'
        '<span class="ltx_note_content"><span class="ltx_note_type">footnotetext: </span>'
        "Accepted at a conference.</span></span></span>"
    )
    doc = _parse(note + section("S1", "1", "Introduction", para("Body.")))
    assert [s.title for s in doc.sections] == ["Introduction"]


def test_a_paper_without_sections_is_split_into_parts() -> None:
    body = "".join(para(prose(f"topic{i}", sentences=20)) for i in range(8))
    doc = _parse(body)
    assert len(doc.sections) > 1
    assert all(s.title.startswith("Part ") for s in doc.sections)


def test_pages_that_are_not_latexml_documents_are_rejected() -> None:
    placeholder = b"<html><body><h1>HTML is not available for the source.</h1></body></html>"
    assert html_unusable_reason(placeholder) is not None
    with pytest.raises(ParseError):
        parse_html(placeholder, "x")
    fatal = latexml_page("<div>Conversion to HTML had a Fatal error and exited abruptly.</div>")
    assert "fatal" in (html_unusable_reason(fatal) or "").lower()
    assert html_unusable_reason(latexml_page(section("S1", "1", "T", para("x")))) is None
    with pytest.raises(ParseError, match="no body text"):
        parse_html(latexml_page(""), "x")


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"),
        ("CC BY-NC-SA 4.0", "https://creativecommons.org/licenses/by-nc-sa/4.0/"),
        ("CC Zero", "https://creativecommons.org/publicdomain/zero/1.0/"),
        (
            "arXiv.org perpetual non-exclusive license",
            "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
        ),
        (None, None),
    ],
)
def test_license_from_the_info_box(label: str | None, expected: str | None) -> None:
    assert html_license(latexml_page("", license_label=label)) == expected


def test_real_latexml_paper() -> None:
    data = gzip.decompress((FIXTURES / "2508.21267v1.html.gz").read_bytes())
    doc = parse_html(data, "2508.21267")
    check_document(doc)
    assert [(s.id, s.title) for s in doc.sections] == [
        ("s1", "Introduction"),
        ("s2", "Background"),
        ("s2.1", "SRM0-RNL Neuron"),
        ("s2.2", "Unary Sorting"),
        ("s3", "Motivation and Opportunity"),
        ("s4", "Catwalk Neuron via Unary Top-K"),
        ("s4.1", "Catwalk Neuron Microarchitecture"),
        ("s4.2", "Unary Top-K"),
        ("s5", "Experimental Setup"),
        ("s6", "Evaluation"),
        ("s6.1", "Gate Count Analysis"),
        ("s6.2", "Synthesis Results"),
        ("s6.2.1", "Unary Top-K"),
        ("s6.2.2", "Dendrite"),
        ("s6.2.3", "Neuron"),
        ("s6.3", "Place and Route Results"),
        ("s7", "Conclusion"),
    ]
    assert doc.sections[0].text.startswith("Neuromorphic computing, particularly represented")
    assert [c.id for c in doc.captions] == [f"fig{i}" for i in range(1, 11)] + ["tab1"]
    assert html_license(data) == "https://creativecommons.org/licenses/by/4.0/"
    text = _all_text(doc)
    assert "$" in text  # math survives as LaTeX
    assert "References" not in [s.title for s in doc.sections]
