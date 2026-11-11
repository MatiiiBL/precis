import gzip
import time

from hypothesis import given
from hypothesis import strategies as st

from precis.models import Document
from precis.text.bm25 import PASSAGE_WORDS, PaperIndex, split_passages, tokenize
from precis.text.document import SectionDraft, build_document
from precis.text.html import parse_html

from .helpers import FIXTURES


def _doc(sections: dict[str, str]) -> Document:
    drafts = [SectionDraft(title, 1, [text]) for title, text in sections.items()]
    return build_document("p", "html", drafts, [], "sha")


FILLER = "The design keeps the pipeline full and the area small in every configuration. "


def test_tokenizer_folds_case_plurals_compounds_and_stopwords() -> None:
    assert tokenize("The KV-Caches of GPUs") == ["kv", "cache", "gpu"]
    assert tokenize("policies and process") == ["policy", "process"]
    assert tokenize("") == []


def test_passages_stay_inside_sections_and_near_the_target_size() -> None:
    doc = _doc({"Intro": FILLER * 40, "Design": FILLER * 3})
    index = PaperIndex(doc)
    assert {p.section_id for p in index.passages} == {"s1", "s2"}
    intro = [p for p in index.passages if p.section_id == "s1"]
    assert [p.id for p in intro] == [f"s1#p{n}" for n in range(1, len(intro) + 1)]
    assert all(PASSAGE_WORDS * 0.5 <= len(p.text.split()) <= PASSAGE_WORDS * 1.5 for p in intro)
    assert " ".join(p.text for p in intro) == (FILLER * 40).strip()


def test_passages_end_at_sentence_boundaries() -> None:
    sentences = [f"Sentence number {n} talks about caches and memory systems." for n in range(60)]
    for passage in split_passages(" ".join(sentences)):
        assert passage.startswith("Sentence number")
        assert passage.endswith("memory systems.")


def test_an_endless_sentence_is_still_cut() -> None:
    passages = split_passages("word " * 1000)
    assert len(passages) > 5
    assert all(len(p.split()) <= PASSAGE_WORDS * 1.5 for p in passages)


@given(st.text(alphabet="ab .\n", max_size=3000))
def test_splitting_keeps_every_word_in_order(text: str) -> None:
    assert " ".join(split_passages(text, words=20)).split() == text.split()


def test_search_ranks_rare_matching_terms_first() -> None:
    doc = _doc(
        {
            "Intro": FILLER * 3 + "We study a prefetcher for irregular graph workloads.",
            "Design": FILLER * 3 + "The prefetcher issues requests to the DRAM controller.",
            "Results": FILLER * 3,
        }
    )
    hits = PaperIndex(doc).search("graph prefetcher", k=5)
    assert [h.section_id for h in hits] == ["s1", "s2"]
    assert hits[0].score > hits[1].score > 0
    assert "graph" in hits[0].text


def test_search_edge_cases() -> None:
    index = PaperIndex(_doc({"Intro": FILLER * 3}))
    assert index.search("quantum entanglement") == []
    assert index.search("the of and") == []
    assert index.search("pipeline", k=0) == []
    assert len(index.search("pipeline", k=1)) == 1
    empty = PaperIndex(build_document("p", "pdf", [], [], "sha"))
    assert empty.search("anything") == []


def test_search_on_a_real_paper() -> None:
    data = gzip.decompress((FIXTURES / "2508.21267v1.html.gz").read_bytes())
    doc = parse_html(data, "2508.21267")
    started = time.perf_counter()
    index = PaperIndex(doc)
    hits = index.search("place and route area power", k=3)
    elapsed = time.perf_counter() - started
    assert hits[0].section_id == "s6.3"  # "Place and Route Results"
    assert elapsed < 0.5
    assert len(index.passages) >= doc.n_tokens * 4 // 5 // 150
