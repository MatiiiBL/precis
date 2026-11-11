"""Group the corpus into topics and name each one.

Clustering is classic latent semantic analysis rather than embeddings: each
summary's tldr, approach, methods and keywords are vectorized with TF-IDF
(sublinear term frequency, unigrams and bigrams), reduced with truncated SVD
and L2-normalized, and k-means runs on the unit vectors, which makes it a
cosine-distance clustering. The summaries are already short, dense and
written in one consistent style, which is where TF-IDF works well, and the
whole step runs in seconds on 2,500 papers with no extra model or service.

k comes from the silhouette score over ``k_min..k_max`` unless it is given.
Every random step takes the configured seed, so the same summaries always
give the same topics, and topic ids are assigned by size (largest first),
not by k-means' arbitrary cluster order.

Each cluster is described by class-based TF-IDF keywords (terms frequent in
the cluster and rare elsewhere) and then named by the model from its most
central papers, with the other clusters' keywords shown for contrast so the
names stay distinct.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from precis.config import AgentConfig, CorpusConfig
from precis.corpus.load import FinishedPaper
from precis.corpus.structured import request_structured, structured_params
from precis.llm.base import ModelGateway
from precis.models import PaperSummary, Topic

logger = logging.getLogger(__name__)

type FloatMatrix = NDArray[np.float64]
type IntVector = NDArray[np.int64]

# Words that occur across research writing regardless of the field. Left in,
# they dominate the TF-IDF vocabulary of summaries ("we propose a novel
# approach that significantly improves ...") and blur every cluster.
_RESEARCH_WRITING_WORDS = """
    paper papers work works study studies propose proposes proposed proposing present presents
    presented introduce introduces introduced approach approaches method methods methodology
    technique techniques framework frameworks novel new existing prior previous recent recently
    result results show shows shown demonstrate demonstrates demonstrated achieve achieves
    achieved achieving improve improves improved improving improvement improvements outperform
    outperforms outperforming significant significantly substantially compared compare comparison
    baseline baselines state art sota use uses used using based leverage leverages leveraging
    enable enables enabling effective effectively efficient efficiently evaluate evaluates
    evaluated evaluation experiment experiments experimental analysis analyze analyzes key main
    various different several overall large small high low better best perform performs
    performing performance task tasks problem problems challenge challenges address addresses
    addressing paradigm design designs designed apply applies applied application applications
    finally furthermore additionally moreover respectively including include includes et al
    author authors
"""
TECHNICAL_STOP_WORDS = frozenset(_RESEARCH_WRITING_WORDS.split())
STOP_WORDS = ENGLISH_STOP_WORDS | TECHNICAL_STOP_WORDS
# Words start with a letter and may contain digits and inner hyphens
# ("3d" is dropped, "network-on-chip" and "gpt-4" are kept whole).
_TOKEN = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)*\b")
_SEGMENT = re.compile(r"[,;:.!?()\[\]\n]+(?:\s|$)|\n")


def analyze(text: str) -> list[str]:
    """Unigrams and bigrams of ``text``, without stop words.

    Bigrams are formed within a clause or list item only, so the keyword
    list "caches, prefetching" does not produce the phrase "caches
    prefetching".
    """
    terms: list[str] = []
    for segment in _SEGMENT.split(text.lower()):
        words = [w for w in _TOKEN.findall(segment) if w not in STOP_WORDS]
        terms += words
        terms += [f"{a} {b}" for a, b in itertools.pairwise(words)]
    return terms


def topic_text(summary: PaperSummary) -> str:
    """The text a paper is clustered on: what it does and with what, not its findings."""
    return "\n".join(
        [summary.tldr, summary.approach, ", ".join(summary.methods), ", ".join(summary.keywords)]
    )


@dataclass(frozen=True)
class Clustering:
    """Cluster assignments for a list of papers, with cluster 0 the largest."""

    paper_ids: list[str]
    labels: IntVector
    vectors: FloatMatrix
    """Unit-length LSA vectors, one row per paper."""
    k: int
    silhouette: dict[int, float]
    """Silhouette score for each k tried; empty when k was given."""

    def members(self, cluster: int) -> list[int]:
        """Row indices of the papers in ``cluster``, most central first."""
        rows = np.flatnonzero(self.labels == cluster)
        centroid = self.vectors[rows].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        similarity = self.vectors[rows] @ centroid
        # Stable sort on negated similarity: ties keep paper-id order.
        return [int(rows[i]) for i in np.argsort(-similarity, kind="stable")]


def _vectorize(texts: Sequence[str], components: int, seed: int) -> FloatMatrix:
    n = len(texts)
    vectorizer = TfidfVectorizer(
        analyzer=analyze,
        sublinear_tf=True,
        min_df=2 if n >= 20 else 1,
        max_df=0.5 if n >= 20 else 1.0,
        dtype=np.float64,
    )
    tfidf = vectorizer.fit_transform(texts)
    n_components = min(components, tfidf.shape[1] - 1, n - 1)
    if n_components < 2:
        dense: FloatMatrix = np.asarray(tfidf.toarray(), dtype=np.float64)
        return np.asarray(normalize(dense), dtype=np.float64)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    reduced = svd.fit_transform(tfidf)
    return np.asarray(normalize(reduced), dtype=np.float64)


def _kmeans(vectors: FloatMatrix, k: int, seed: int) -> IntVector:
    model = KMeans(n_clusters=k, n_init=8, random_state=seed)
    return np.asarray(model.fit_predict(vectors), dtype=np.int64)


def _order_by_size(labels: IntVector, paper_ids: Sequence[str]) -> IntVector:
    """Renumber clusters by size (largest first), breaking ties by first paper id."""
    clusters = np.unique(labels)

    def key(c: np.int64) -> tuple[int, str]:
        rows = np.flatnonzero(labels == c)
        return (-len(rows), min(paper_ids[int(i)] for i in rows))

    order = sorted(clusters, key=key)
    mapping = {int(old): new for new, old in enumerate(order)}
    return np.array([mapping[int(c)] for c in labels], dtype=np.int64)


def cluster_papers(
    paper_ids: Sequence[str],
    texts: Sequence[str],
    *,
    k: int | None = None,
    k_min: int = 6,
    k_max: int = 24,
    svd_components: int = 100,
    seed: int = 0,
) -> Clustering:
    """Cluster ``texts`` into ``k`` groups, or into the k in ``k_min..k_max`` that
    maximizes the cosine silhouette score.

    The candidate range is clipped to what the corpus can support (at least
    two papers per cluster on average), and a corpus too small to split forms
    a single cluster.
    """
    if len(paper_ids) != len(texts):
        raise ValueError("paper_ids and texts must have the same length")
    n = len(texts)
    if n == 0:
        raise ValueError("cannot cluster an empty corpus")
    vectors = _vectorize(texts, svd_components, seed)
    scores: dict[int, float] = {}
    if k is not None:
        chosen = min(k, n)
        labels = _kmeans(vectors, chosen, seed) if chosen > 1 else np.zeros(n, dtype=np.int64)
    else:
        upper = min(k_max, n // 2)
        lower = min(k_min, upper)
        candidates: dict[int, IntVector] = {}
        for candidate in range(max(lower, 2), upper + 1):
            found = _kmeans(vectors, candidate, seed)
            if len(np.unique(found)) < 2:
                continue
            candidates[candidate] = found
            scores[candidate] = float(
                silhouette_score(vectors, found, metric="cosine", random_state=seed)
            )
        if scores:
            # Highest score wins; on a tie, the smaller k.
            chosen = max(scores, key=lambda c: (scores[c], -c))
            labels = candidates[chosen]
        else:
            chosen, labels = 1, np.zeros(n, dtype=np.int64)
    labels = _order_by_size(labels, paper_ids)
    return Clustering(list(paper_ids), labels, vectors, int(labels.max()) + 1, scores)


def _add_term(term: str, chosen: list[str]) -> None:
    """Add ``term`` to ``chosen`` unless a chosen phrase already contains it.

    A phrase takes the place of the single words it is made of, at the
    position of the best-ranked one: "cache replacement" arriving after
    "cache" and "replacement" replaces both, since unigrams almost always
    outscore the bigrams that contain them.
    """
    words = set(term.split())
    if any(words <= set(other.split()) for other in chosen):
        return
    parts = [i for i, other in enumerate(chosen) if set(other.split()) < words]
    if not parts:
        chosen.append(term)
        return
    chosen[parts[0]] = term
    for i in reversed(parts[1:]):
        del chosen[i]


def class_keywords(texts: Sequence[str], labels: IntVector, *, n: int = 10) -> list[list[str]]:
    """Top ``n`` class-based TF-IDF terms for each cluster.

    All the texts of a cluster are treated as one document: a term scores
    ``tf(term, cluster) * log(1 + A / f(term))``, where ``tf`` is its share of
    the cluster's words, ``A`` the average number of words per cluster and
    ``f`` the term's frequency in the whole corpus. A bigram replaces the
    chosen unigrams it contains, and a unigram inside a chosen bigram is
    skipped, so no keyword repeats another.
    """
    vectorizer = CountVectorizer(analyzer=analyze)
    counts = vectorizer.fit_transform(texts)
    vocabulary = vectorizer.get_feature_names_out()
    k = int(labels.max()) + 1
    one_hot = np.zeros((k, len(texts)), dtype=np.float64)
    one_hot[labels, np.arange(len(texts))] = 1.0
    per_class = np.asarray((counts.T @ one_hot.T).T, dtype=np.float64)
    words_per_class = per_class.sum(axis=1, keepdims=True)
    tf = per_class / np.maximum(words_per_class, 1.0)
    term_freq = per_class.sum(axis=0)
    idf = np.log1p(per_class.sum() / k / np.maximum(term_freq, 1.0))
    scores = tf * idf
    keywords: list[list[str]] = []
    for row in scores:
        chosen: list[str] = []
        for index in np.argsort(-row, kind="stable"):
            if row[index] <= 0 or len(chosen) == n:
                break
            _add_term(str(vocabulary[index]), chosen)
        keywords.append(chosen)
    return keywords


# --------------------------------------------------------------------------
# Naming clusters
# --------------------------------------------------------------------------


class TopicLabel(BaseModel):
    """What the model writes for one cluster."""

    label: str = Field(description="A specific name for the topic, 2-6 words, in title case.")
    description: str = Field(
        description="Two sentences: what the papers in this topic study and how."
    )


LABEL_SYSTEM = """\
You name clusters of research papers for a map of a research field. A clustering \
algorithm grouped the papers; you see the cluster's distinctive keywords, its most \
central papers, and the keywords of the other clusters.

Give the cluster a specific name that a researcher in the field would recognize, \
that fits all the papers shown, and that distinguishes it from the other clusters. \
Avoid generic names such as "Machine Learning Methods" or "Hardware Design". Then \
describe the topic in two sentences: what problems its papers study and which \
approaches they take. Do not mention the clustering, the keywords or individual \
papers."""


@dataclass(frozen=True)
class ClusterBrief:
    """What the model is shown to name one cluster."""

    size: int
    keywords: list[str]
    central: list[FinishedPaper]
    other_keywords: list[list[str]]


def label_prompt(brief: ClusterBrief) -> str:
    lines = [
        f"Cluster size: {brief.size} papers",
        f"Distinctive keywords: {', '.join(brief.keywords)}",
        "",
        "Most central papers:",
    ]
    for paper in brief.central:
        lines.append(f"- {paper.meta.title}")
        lines.append(f"  Summary: {paper.summary.tldr}")
        lines.append(f"  Keywords: {', '.join(paper.summary.keywords)}")
    lines += ["", "Other clusters in the corpus (keywords):"]
    lines += [f"- {', '.join(words[:6])}" for words in brief.other_keywords]
    return "\n".join(lines)


def label_params(brief: ClusterBrief, agent: AgentConfig) -> dict[str, Any]:
    return structured_params(
        agent, system=LABEL_SYSTEM, user=label_prompt(brief), output=TopicLabel
    )


class TopicModel(BaseModel):
    """The stored result of ``precis topics``: the topics and how k was chosen."""

    k: int
    seed: int
    silhouette: dict[int, float] = Field(description="Score per k tried; empty when k was given.")
    topics: list[Topic] = Field(description="Ordered by size; paper ids most central first.")


TOPICS_ARTIFACT = "corpus/topics"


async def build_topics(
    gw: ModelGateway,
    papers: Sequence[FinishedPaper],
    config: CorpusConfig,
    agent: AgentConfig,
    *,
    k: int | None = None,
) -> TopicModel:
    """Cluster ``papers`` and name every cluster (one request each, concurrently)."""
    ordered = sorted(papers, key=lambda p: p.id)
    texts = [topic_text(p.summary) for p in ordered]
    clustering = cluster_papers(
        [p.id for p in ordered],
        texts,
        k=k if k is not None else config.k,
        k_min=config.k_min,
        k_max=config.k_max,
        svd_components=config.svd_components,
        seed=config.seed,
    )
    keywords = class_keywords(texts, clustering.labels)
    members = [clustering.members(c) for c in range(clustering.k)]
    briefs = [
        ClusterBrief(
            size=len(rows),
            keywords=keywords[c],
            central=[ordered[i] for i in rows[: config.central_papers]],
            other_keywords=[words for other, words in enumerate(keywords) if other != c],
        )
        for c, rows in enumerate(members)
    ]
    labels = await asyncio.gather(
        *(
            request_structured(gw, label_params(brief, agent), TopicLabel, stage="label")
            for brief in briefs
        )
    )
    topics = [
        Topic(
            id=c + 1,
            label=label.label.strip(),
            description=label.description.strip(),
            keywords=keywords[c],
            paper_ids=[ordered[i].id for i in members[c]],
        )
        for c, label in enumerate(labels)
    ]
    logger.info("built %d topics over %d papers", len(topics), len(ordered))
    return TopicModel(
        k=clustering.k, seed=config.seed, silhouette=clustering.silhouette, topics=topics
    )


def topic_of(topics: Sequence[Topic]) -> dict[str, Topic]:
    """Map each paper id to its topic."""
    return {paper_id: topic for topic in topics for paper_id in topic.paper_ids}
