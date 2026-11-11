import pytest
from hypothesis import given
from hypothesis import strategies as st

from precis.text.document import (
    SectionDraft,
    build_document,
    split_into_parts,
    strip_acknowledgment,
)
from precis.text.normalize import clean_chars, fix_surrogates, join_paragraphs, squash
from precis.util import estimate_tokens


def _ids(drafts: list[SectionDraft]) -> list[tuple[str, int]]:
    doc = build_document("p", "html", drafts, [], "sha")
    return [(s.id, s.level) for s in doc.sections]


def test_ids_follow_the_outline_and_appendices_count_separately() -> None:
    drafts = [
        SectionDraft("Intro", 1, ["a"]),
        SectionDraft("Background", 1, ["b"]),
        SectionDraft("Caches", 2, ["c"]),
        SectionDraft("Policies", 3, ["d"]),
        SectionDraft("Prefetch", 2, ["e"]),
        SectionDraft("Design", 1, ["f"]),
        SectionDraft("Proofs", 1, ["g"], appendix=True),
        SectionDraft("Lemma", 2, ["h"], appendix=True),
    ]
    assert _ids(drafts) == [
        ("s1", 1),
        ("s2", 1),
        ("s2.1", 2),
        ("s2.1.1", 3),
        ("s2.2", 2),
        ("s3", 1),
        ("a1", 1),
        ("a1.1", 2),
    ]


def test_levels_are_normalized_so_every_id_has_a_parent() -> None:
    # A paper that only uses \subsection, then jumps two levels down.
    drafts = [
        SectionDraft("One", 2, ["a"]),
        SectionDraft("Deep", 4, ["b"]),
        SectionDraft("Two", 2, ["c"]),
    ]
    assert _ids(drafts) == [("s1", 1), ("s1.1", 2), ("s2", 1)]


def test_empty_leaves_are_dropped_but_empty_parents_kept() -> None:
    drafts = [
        SectionDraft("Evaluation", 1, []),
        SectionDraft("Setup", 2, ["text"]),
        SectionDraft("Empty", 1, []),
        SectionDraft("Conclusion", 1, ["end"]),
    ]
    doc = build_document("p", "pdf", drafts, [], "sha")
    assert [(s.id, s.title) for s in doc.sections] == [
        ("s1", "Evaluation"),
        ("s1.1", "Setup"),
        ("s2", "Conclusion"),
    ]
    assert doc.n_tokens == sum(s.n_tokens for s in doc.sections)
    assert doc.sections[0].n_tokens == 0


def test_section_text_is_normalized_paragraphs() -> None:
    doc = build_document("p", "html", [SectionDraft("T", 1, ["  a\n b  ", "", "c"])], [], "x")
    assert doc.sections[0].text == "a b\n\nc"
    assert doc.sections[0].n_tokens == estimate_tokens("a b\n\nc")


def test_split_into_parts_respects_the_target_and_paragraphs() -> None:
    paragraphs = [f"Paragraph {i}. " + "word " * 300 for i in range(12)]
    parts = split_into_parts(paragraphs, target_tokens=1500)
    assert [p.title for p in parts] == [f"Part {i}" for i in range(1, len(parts) + 1)]
    assert all(estimate_tokens(p.text) <= 1500 + 10 for p in parts)
    assert len(parts) > 1
    # Whole paragraphs stay whole when they fit.
    assert all(p.startswith("Paragraph") for part in parts for p in part.paragraphs)


def test_a_paragraph_longer_than_a_part_is_cut_between_sentences() -> None:
    sentence = "This sentence about cache coherence is long enough to matter here. "
    parts = split_into_parts([sentence * 200], target_tokens=500)
    assert len(parts) > 1
    for part in parts:
        assert part.paragraphs[0].startswith("This sentence")
        assert part.paragraphs[-1].endswith("matter here.")


@given(st.lists(st.text(alphabet="abc .\n", max_size=400), max_size=20))
def test_split_into_parts_never_loses_or_reorders_words(paragraphs: list[str]) -> None:
    parts = split_into_parts(paragraphs, target_tokens=50)
    words = [w for p in paragraphs for w in p.split()]
    assert [w for part in parts for p in part.paragraphs for w in p.split()] == words


def test_surrogates_are_recombined_or_dropped() -> None:
    """2512.14151: pypdf returned math letters as lone UTF-16 halves ('\\ud835')."""
    high, low = chr(0xD835), chr(0xDC4E)  # the two halves of U+1D44E, italic "a"
    assert fix_surrogates(f"x {high}{low} y") == f"x {chr(0x1D44E)} y"
    assert fix_surrogates(f"S = {{{high}1, {high}2}}") == "S = {1, 2}"
    assert fix_surrogates(f"{low}lone low") == "lone low"
    assert fix_surrogates("plain text") == "plain text"
    # clean_chars, which every parser uses, also plain-letters the result.
    assert clean_chars(f"{high}{low}{high}") == "a"
    clean_chars(f"{high}x").encode("utf-8")  # encodable: no surrogate left


@pytest.mark.parametrize(
    ("paragraph", "kept"),
    [
        # 2507.10912: a bold run-in at the start of the last paragraph.
        (
            "Acknowledgment This work is sponsored by the National Science Foundation "
            "under Grant No. CCF2403134.",
            "",
        ),
        ("Our results hold. Acknowledgments: We thank the reviewers.", "Our results hold."),
        ("Acknowledgements. The authors thank X.", ""),
        ("ACKNOWLEDGMENT The authors thank X.", ""),
        # Not acknowledgments: the word inside a sentence, or another word.
        ("Acknowledging the limits, we restrict the study.", None),
        ("As the acknowledgments note, Section 3 relies on it.", None),
        ("We list grants in the Acknowledgments. That is all.", None),
    ],
)
def test_run_in_acknowledgments_are_cut(paragraph: str, kept: str | None) -> None:
    assert strip_acknowledgment(paragraph) == (paragraph if kept is None else kept)


def test_build_document_drops_acknowledgment_paragraphs() -> None:
    drafts = [
        SectionDraft("Conclusion", 1, ["We conclude.", "Acknowledgment This work is funded."])
    ]
    doc = build_document("p", "html", drafts, [], "sha")
    assert doc.sections[0].text == "We conclude."


def test_clean_chars_expands_ligatures_and_math_letters() -> None:
    ffi, zero_width, nbsp = chr(0xFB03), chr(0x200B), chr(0x00A0)
    math_sec = chr(0x1D460) + chr(0x1D452) + chr(0x1D450)  # italic "sec"
    assert clean_chars(f"e{ffi}cient {math_sec}{zero_width}") == "efficient sec"
    assert squash(f" a{nbsp}\n\tb ") == "a b"
    assert join_paragraphs(["x  y", " ", "z"]) == "x y\n\nz"
