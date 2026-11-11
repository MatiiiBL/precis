from collections import Counter
from typing import Any

import numpy as np
import pytest

from corpus.factories import THEMES, ScriptedGateway, make_view, themed_corpus
from precis.config import AgentConfig, CorpusConfig
from precis.corpus.topics import (
    TopicLabel,
    analyze,
    build_topics,
    class_keywords,
    cluster_papers,
    topic_text,
)


def _purity(labels: np.ndarray, ids: list[str], truth: dict[str, str]) -> float:
    """Share of papers whose cluster's majority theme is their own theme."""
    agree = 0
    for cluster in np.unique(labels):
        themes = Counter(truth[ids[i]] for i in np.flatnonzero(labels == cluster))
        agree += themes.most_common(1)[0][1]
    return agree / len(ids)


def test_silhouette_recovers_the_planted_themes() -> None:
    papers, truth = themed_corpus(30)
    ids = [p.id for p in papers]
    clustering = cluster_papers(ids, [topic_text(p.summary) for p in papers], k_min=2, k_max=8)
    assert clustering.k == len(THEMES)
    assert set(clustering.silhouette) == set(range(2, 9))
    assert _purity(clustering.labels, ids, truth) == 1.0


def test_clusters_are_numbered_by_size_and_deterministic() -> None:
    papers, _ = themed_corpus(20)
    papers = papers[:-15]  # the last theme becomes the smallest cluster
    ids = [p.id for p in papers]
    texts = [topic_text(p.summary) for p in papers]
    first = cluster_papers(ids, texts, k=4, seed=7)
    again = cluster_papers(ids, texts, k=4, seed=7)
    assert np.array_equal(first.labels, again.labels)
    sizes = [int((first.labels == c).sum()) for c in range(first.k)]
    assert sizes == sorted(sizes, reverse=True)
    assert first.silhouette == {}


def test_members_are_ordered_by_centrality() -> None:
    papers, _ = themed_corpus(10)
    ids = [p.id for p in papers]
    clustering = cluster_papers(ids, [topic_text(p.summary) for p in papers], k=4)
    for cluster in range(clustering.k):
        rows = clustering.members(cluster)
        centroid = clustering.vectors[rows].mean(axis=0)
        similarity = clustering.vectors[rows] @ (centroid / np.linalg.norm(centroid))
        assert list(similarity) == sorted(similarity, reverse=True)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_tiny_corpora_form_one_cluster(n: int) -> None:
    papers, _ = themed_corpus(1)
    papers = papers[:n]
    clustering = cluster_papers([p.id for p in papers], [topic_text(p.summary) for p in papers])
    assert clustering.k == 1
    assert list(clustering.labels) == [0] * n


def test_given_k_larger_than_corpus_is_clipped() -> None:
    papers, _ = themed_corpus(1)
    clustering = cluster_papers(
        [p.id for p in papers], [topic_text(p.summary) for p in papers], k=50
    )
    assert clustering.k == len(papers)


def test_cluster_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        cluster_papers([], [])
    with pytest.raises(ValueError, match="same length"):
        cluster_papers(["a"], ["x", "y"])


def test_class_keywords_are_distinctive_and_not_redundant() -> None:
    papers, truth = themed_corpus(15)
    ids = [p.id for p in papers]
    texts = [topic_text(p.summary) for p in papers]
    clustering = cluster_papers(ids, texts, k=4)
    keywords = class_keywords(texts, clustering.labels, n=6)
    for cluster, words in enumerate(keywords):
        theme = Counter(truth[ids[i]] for i in np.flatnonzero(clustering.labels == cluster))
        vocabulary = " ".join(THEMES[theme.most_common(1)[0][0]])
        assert len(words) == 6
        assert all(w in vocabulary for w in words[:3]), words
        for i, word in enumerate(words):
            for other in words[i + 1 :]:
                assert not set(other.split()) <= set(word.split())
                assert not set(word.split()) <= set(other.split())


def test_bigrams_stay_within_list_items() -> None:
    terms = analyze("We propose cache replacement, prefetching; a network-on-chip.\nGPT-4 wins")
    assert terms == [
        "cache",
        "replacement",
        "cache replacement",
        "prefetching",
        "network-on-chip",
        "gpt-4",
        "wins",
        "gpt-4 wins",
    ]


def test_latex_markup_is_not_vocabulary() -> None:
    terms = analyze(r"Mitigation with $\mathrm{PRAC}$ costs $O(n^2)$, \textbf{DRAM timing}")
    assert "mathrm" not in " ".join(terms)
    assert "textbf" not in " ".join(terms)
    assert {"prac", "dram timing", "mitigation"} <= set(terms)
    assert all(len(term) > 1 for term in terms)


def test_small_corpora_are_not_split_into_singletons() -> None:
    papers, _ = themed_corpus(11)  # 44 papers, four themes
    ids = [p.id for p in papers]
    clustering = cluster_papers(ids, [topic_text(p.summary) for p in papers], k_min=2, k_max=24)
    assert max(clustering.silhouette) <= 7  # about sqrt(44)
    assert np.bincount(clustering.labels).min() >= 3


def test_phrases_replace_the_words_they_contain() -> None:
    texts = ["cache replacement policy"] * 3 + ["qubit routing"] * 3
    keywords = class_keywords(texts, np.array([0, 0, 0, 1, 1, 1]))
    assert keywords[1] == ["qubit routing"]
    assert set(keywords[0]) == {"cache replacement", "replacement policy"}


def test_technical_stop_words_are_not_keywords() -> None:
    texts = ["we propose a novel cache approach"] * 3 + ["we propose a novel qubit approach"] * 3
    keywords = class_keywords(texts, np.array([0, 0, 0, 1, 1, 1]))
    assert keywords == [["cache"], ["qubit"]]


async def test_build_topics_names_each_cluster_from_central_papers() -> None:
    papers, _ = themed_corpus(12)

    def script(params: dict[str, Any], stage: str) -> dict[str, str]:
        assert stage == "label"
        assert params["output_config"]["format"]["type"] == "json_schema"
        prompt = params["messages"][0]["content"]
        theme = next(t for t in sorted(THEMES) if f"{t.title()} paper" in prompt)
        return {"label": f" {theme.title()} ", "description": f"Papers on {theme}. "}

    gateway = ScriptedGateway(script)
    config = CorpusConfig(k=4, central_papers=3)
    model = await build_topics(gateway, papers, config, AgentConfig(effort="high"))

    assert model.k == 4
    assert gateway.stages() == ["label"] * 4
    assert sorted(t.label for t in model.topics) == sorted(t.title() for t in THEMES)
    assert all(t.description.endswith(".") and t.keywords for t in model.topics)
    assert [t.id for t in model.topics] == [1, 2, 3, 4]
    assert sorted(p for t in model.topics for p in t.paper_ids) == sorted(p.id for p in papers)
    # Only the three most central papers are shown, plus every other cluster.
    prompt = gateway.calls[0].user_text
    assert prompt.count("Summary:") == 3
    assert prompt.split("Other clusters")[1].count("\n- ") == 3


async def test_build_topics_requests_are_deterministic() -> None:
    papers, _ = themed_corpus(6)
    gateways = []
    for order in (papers, list(reversed(papers))):
        gateway = ScriptedGateway(lambda p, s: TopicLabel(label="X", description="Y").model_dump())
        await build_topics(gateway, order, CorpusConfig(k=4), AgentConfig())
        gateways.append(gateway)
    first, second = ([c.params for c in g.calls] for g in gateways)
    assert first == second


async def test_build_topics_uses_the_k_override() -> None:
    papers = [make_view(f"c{i}", ["cache replacement", "prefetching", f"llc{i}"]) for i in range(4)]
    papers += [make_view(f"q{i}", ["qubit routing", "surface code", f"gate{i}"]) for i in range(4)]
    gateway = ScriptedGateway(lambda p, s: {"label": "Caches", "description": "D."})
    model = await build_topics(gateway, papers, CorpusConfig(), AgentConfig(), k=2)
    assert model.k == 2
    assert model.silhouette == {}
