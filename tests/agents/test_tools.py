import pytest

from precis.agents.paper import PaperView
from precis.agents.tools import ReadingTools, paper_index
from precis.models import Caption

from .helpers import DESIGN_1, DESIGN_2, RESULTS, make_doc, make_meta


@pytest.fixture
def tools() -> ReadingTools:
    doc = make_doc()
    return ReadingTools(
        PaperView(make_meta(), doc, page_tokens=40), paper_index(doc), search_results=2
    )


def test_read_section_pages_through_a_section(tools: ReadingTools) -> None:
    first = tools.run("read_section", {"section_id": "s3", "page": 1})
    assert not first.is_error
    assert first.text.startswith("[s3] Design of ReuseNet (page 1 of 2)\n\n")
    assert DESIGN_1 in first.text
    assert first.text.endswith("Continues on page 2 of 2.")
    second = tools.run("read_section", {"section_id": "s3", "page": 2})
    assert DESIGN_2 in second.text
    assert second.text.endswith("End of section s3. Next in the outline: s4 Evaluation.")
    last = tools.run("read_section", {"section_id": "s5", "page": 1})
    assert last.text.endswith("the last section of the body.")


def test_rereading_a_page_returns_a_pointer_not_the_text(tools: ReadingTools) -> None:
    tools.run("read_section", {"section_id": "s1", "page": 1})
    tokens = tools.log.tokens_read
    again = tools.run("read_section", {"section_id": "s1", "page": 1})
    assert "already read page 1 of section s1" in again.text
    assert tools.log.tokens_read == tokens


def test_unknown_ids_and_pages_are_errors_that_list_the_choices(tools: ReadingTools) -> None:
    unknown = tools.run("read_section", {"section_id": "s4.3", "page": 1})
    assert unknown.is_error
    assert "Did you mean" in unknown.text
    assert "Section ids: s1, s2, s2.1, s3, s4, s4.1, s4.2, s5." in unknown.text
    assert "Caption ids: fig1, tab1." in unknown.text
    page = tools.run("read_section", {"section_id": "s3", "page": 3})
    assert page.is_error
    assert page.text == "Section s3 has 2 pages; page 3 does not exist. Pages are numbered from 1."
    zero = tools.run("read_section", {"section_id": "s1", "page": 0})
    assert zero.is_error


def test_heading_only_sections_point_to_their_subsections(tools: ReadingTools) -> None:
    out = tools.run("read_section", {"section_id": "s4", "page": 1})
    assert not out.is_error
    assert out.text == (
        "Section s4 (Evaluation) has no text of its own. "
        "Read its subsections instead: s4.1 Setup, s4.2 Results."
    )


def test_caption_ids_return_the_full_caption(tools: ReadingTools) -> None:
    out = tools.run("read_section", {"section_id": " tab1 ", "page": 1})
    assert out.text == "[tab1] (table)\n\nStorage: ReuseNet 12.5 KB; Hawkeye 28 KB."


def test_search(tools: ReadingTools) -> None:
    out = tools.run("search_paper", {"query": "IPC over Hawkeye"})
    assert out.text.startswith("2 passages for 'IPC over Hawkeye', best first:")
    # A short section is a single BM25 passage.
    assert f"1. [s4.2] Results\n{RESULTS}" in out.text
    assert out.text.endswith("read_section shows a passage in its context.")
    assert tools.run("search_paper", {"query": "  "}).is_error
    empty = tools.run("search_paper", {"query": "quantum"})
    assert not empty.is_error
    assert empty.text.startswith("No passage matches 'quantum'.")


def test_captions_are_listed_and_shortened_when_long() -> None:
    doc = make_doc()
    long = Caption(id="tab2", kind="table", text="cell " * 2000)
    doc = doc.model_copy(update={"captions": [*doc.captions, long]})
    tools = ReadingTools(
        PaperView(make_meta(), doc, page_tokens=1000), paper_index(doc), search_results=2
    )
    out = tools.run("read_captions", {})
    assert out.text.startswith("[fig1] (figure) Miss reduction over LRU")
    assert "[tab1] (table) Storage: ReuseNet 12.5 KB; Hawkeye 28 KB." in out.text
    assert out.text.endswith("[shortened]")
    assert len(out.text) < 2000
    none = ReadingTools(
        PaperView(make_meta(), doc.model_copy(update={"captions": []}), page_tokens=40),
        paper_index(doc),
        search_results=2,
    )
    assert none.run("read_captions", {}).text == "This paper has no figure or table captions."


def test_unknown_tool_is_an_error(tools: ReadingTools) -> None:
    out = tools.run("read_everything", {})
    assert out.is_error


def test_fraction_read_counts_unique_pages(tools: ReadingTools) -> None:
    doc = tools.view.doc
    assert tools.log.fraction_of(doc) == 0
    tools.run("read_section", {"section_id": "s3", "page": 1})
    tools.run("read_section", {"section_id": "s3", "page": 2})
    assert tools.log.pages == {("s3", 1), ("s3", 2)}
    assert 0 < tools.log.fraction_of(doc) < 0.5
