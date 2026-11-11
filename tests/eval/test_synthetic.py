import statistics

from precis.eval.synthetic import synthetic_corpus, synthetic_paper


def test_papers_are_reproducible_and_distinct() -> None:
    assert synthetic_paper(7, seed=1) == synthetic_paper(7, seed=1)
    assert synthetic_paper(7, seed=1) != synthetic_paper(7, seed=2)
    corpus = synthetic_corpus(50)
    assert len({meta.id for meta, _ in corpus}) == 50
    assert len({meta.title for meta, _ in corpus}) > 40


def test_papers_have_the_shape_the_agents_read() -> None:
    meta, doc = synthetic_paper(3)
    titles = [s.title for s in doc.sections]
    assert titles[0] == "Introduction"
    assert "Evaluation" in titles
    assert "Discussion and Limitations" in titles
    assert doc.paper_id == meta.id
    assert doc.n_tokens == sum(s.n_tokens for s in doc.sections)
    assert "We propose" in doc.sections[0].text
    assert "%" in meta.abstract
    assert meta.abstract not in " ".join(s.text for s in doc.sections)


def test_body_lengths_follow_real_papers() -> None:
    lengths = [doc.n_tokens for _, doc in synthetic_corpus(400)]
    assert 7000 < statistics.median(lengths) < 10000
    assert max(lengths) > 2 * min(lengths)
