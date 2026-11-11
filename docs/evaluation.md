# Corpus outputs, evaluation and benchmarks

This document covers the packages that run after the per-paper pipeline
(`precis.corpus`, `precis.eval`): turning a few thousand verified summaries
into topics, a written report and a static site, and measuring how good the
summaries are and how fast the pipeline runs. Section 6 has the numbers.

| Command | What it does | Requests |
|---|---|---|
| `precis topics [--k N]` | cluster the summaries, name each topic | one per topic |
| `precis report [--out PATH]` | synthesize each topic, write the introduction, render Markdown | one per topic (more for large topics), plus one |
| `precis site [--out DIR]` | render the static site | none |
| `precis eval abstract [--n N]` | summaries written without the abstract vs the abstract | reader runs plus two judge requests per paper |
| `precis eval faithfulness` | verdict rates before and after revision, quote audit | none |
| `precis eval ablation [--n N]` | agentic vs fulltext vs mapreduce on the same papers | three reader runs, a critic run and judge requests per paper |
| `precis bench throughput` | the full pipeline offline over 2,000 papers, across concurrency ceilings | simulated |
| `precis bench faults` | injected API errors and a crash mid-run, then a resume | simulated |

All model requests go through `ModelGateway`; every command accepts
`--offline`, which answers with deterministic stand-ins
(`precis.corpus.offline`, `precis.eval.offline`), so the whole path runs in
CI without an API key.

## 1. Topics

**Text.** A paper is clustered on its summary's tldr, approach, methods and
keywords: what it does and with what. Findings are left out on purpose; they
are mostly numbers and benchmark names, which group papers by evaluation
setup rather than by subject.

**Vectors.** TF-IDF with sublinear term frequency over unigrams and bigrams,
then truncated SVD to 100 dimensions and L2 normalization (classic latent
semantic analysis). Two details matter more than the choice of algorithm:

- *Stop words.* Besides English stop words, about 150 words of research
  prose ("propose", "novel", "outperform", "significantly", "framework",
  "performance") are removed. In LLM-written summaries these are the most
  frequent content words, and left in they pull every cluster towards the
  same centre.
- *Phrases.* Bigrams are formed only inside a clause or list item, so the
  keyword list "cache replacement, prefetching" yields "cache replacement"
  but not "replacement prefetching". Without this rule, phrases that no
  paper uses, such as "replacement memory", made it into the topic
  keywords.

Embeddings would be the other option. LSA was chosen because the summaries
are short, dense and written in one consistent style (which is where TF-IDF
works well), because it needs no second model, service or GPU, and because
it is fast and exactly reproducible: the same summaries and seed give the
same topics, which keeps the response cache valid for the naming requests.

**Clustering.** k-means on the unit vectors (that is, cosine k-means), with
`n_init=8`. k is the value in `corpus.k_min..k_max` (default 6..24, clipped
to at most n/2) with the highest cosine silhouette score, unless `--k` or
`corpus.k` gives it. Topic ids are assigned by size, largest first, and
each topic's papers are ordered by cosine similarity to the centroid, so
"the most representative papers" is simply the head of the list.

**Keywords.** Class-based TF-IDF: each cluster's texts are treated as one
document and a term scores `tf(term, cluster) * log(1 + A / f(term))`, where
`A` is the average number of words per cluster and `f` the term's corpus
frequency. Plain TF-IDF over papers would reward terms that are rare
everywhere; this rewards terms frequent in the cluster and rare elsewhere.
A bigram replaces the unigrams it contains ("cache replacement" instead of
"cache" and "replacement"), since unigrams almost always outscore the
phrases that contain them.

**Naming.** One structured-output request per cluster (role
`agents.synthesizer`), showing the cluster's keywords, its eight most
central papers (title, tldr, keywords) and the keywords of every other
cluster. The other clusters are there for contrast: without them, sibling
clusters in one field get near-identical names.

**Measured** (Apple M4 laptop, 2,500 synthetic summaries from 12
overlapping themes): the silhouette search over k = 6..24 takes 1.5 s and
picks k = 11 (silhouette 0.266, tied with 13); a fixed k takes 0.12 s;
keywords take 0.05 s. On a corpus with four planted, disjoint themes the
search recovers k = 4 with cluster purity 1.0 (`tests/corpus/test_topics.py`).

**Caveat.** The silhouette score measures separation, not usefulness, and
it is often flat: in the run above every k from 10 to 24 scored within
0.025 of the best. Treat the chosen k as a reasonable default and pass
`--k` when you know the granularity you want.

## 2. Synthesis and report

Each topic gets one request that writes a `TopicSynthesis`: a 3-5 paragraph
overview, trends, open problems and key papers, citing papers as
`[arXiv:<id>]`. The model sees compact summaries (tag, title, tldr and
findings; about 240 tokens for a summary with a 40-word tldr and four
findings), not full summaries: that is what a synthesis needs, and it lets
a topic of about 250 papers fit in one request.

**Fitting the context window.** `corpus.synthesis_input_tokens` (60,000 by
default) caps the paper text in one request, far below the model's 1M
context: long inputs cost more, take longer and get less even attention. A
larger topic is never truncated. Its papers are dealt round-robin, in
centrality order, into the fewest chunks that fit, so every chunk is a
cross-section of the topic rather than its most or least typical slice;
each chunk gets a partial synthesis, and the partials are merged, in groups
and recursively if they do not fit together. A 1,000-paper topic at the
default budget takes four partial requests and one merge. A single paper
larger than the budget is an error, not a cut.

**Citations.** After each request, every citation is checked against the
topic's papers. A citation of a paper outside the topic (a hallucinated id,
or a paper from a neighbouring topic) is removed, and the counts are stored
with the report and printed in its closing section. Grouped citations
(`[arXiv:a, arXiv:b]`) are split, since models produce them despite the
instruction. Key papers outside the topic are dropped; if none is left, the
three most central papers are used.

**Introduction.** A last request writes the title and introduction from the
corpus statistics and each topic's name, size, description and trends (with
citations stripped, since the introduction does not cite).

**Rendering.** `precis report` writes one Markdown file with a table of
contents whose links use GitHub's heading anchors, a numbered section per
topic with citations linked to arXiv, and a closing section with
faithfulness, citation and spend figures.

**Requests.** All corpus and judge requests are single-shot structured
outputs (`output_config.format` with a schema generated from a pydantic
model, adaptive thinking, the role's `effort`). They carry no
`cache_control`: each has a unique body after a short system prompt that is
below the minimum cacheable prefix, so a breakpoint would only add the
cache-write premium. Prompts are deterministic (sorted inputs, no
timestamps), so re-running `precis topics` or `precis report` on unchanged
summaries is served entirely from the gateway's response cache
(`tests/corpus/test_cli.py` checks this).

## 3. Site

`precis site --out build/site` renders, with Jinja2:

- `index.html`: corpus facts, faithfulness before and after revision (a
  stacked bar per stage), the introduction, topic cards, and spend by stage;
- `topics/<id>.html`: the synthesis with linked citations, key papers, and
  every paper in centrality order;
- `papers/<id>.html`: the summary, with the critic's verdict as a badge on
  the tldr and on every finding and the quoted evidence under it, plus
  limitations, methods, datasets, keywords and arXiv links;
- `search.html`: client-side search over every paper's title, tldr,
  keywords, authors and topic.

There are no external scripts, fonts or stylesheets, and the site works
from a `file://` URL. That is why the search index is a JSON document
assigned to a global in `assets/search-index.js` rather than a `.json`
file: browsers refuse `fetch` on `file://` pages. Only the search page
loads it; every other page has a plain form that opens the search page, and
no JavaScript at all. Model-written Markdown is rendered with raw HTML
disabled, so a stray `<script>` in a synthesis shows up as text. Colors
follow `prefers-color-scheme`, and the layout collapses to one column on
phones.

**Measured** with 2,500 papers: 2,504 pages and 10.3 MB written in 0.53 s;
a paper page is about 3.5 KB; the search index is 765 KB (about 300 bytes
per paper), and a two-word query scans all 2,500 entries in 0.7 ms in
Orca's built-in browser (Chromium), so search runs on every keystroke
without an inverted index.

## 4. Evaluation methodology

### 4.1 Summaries without the abstract (`precis eval abstract`)

The reader normally sees the abstract. For this evaluation it does not
(`include_abstract = false`), and the authors' abstract becomes an
independent reference: a summary written from the body alone should
recover what the authors say their paper is about. `--n` papers (default
`eval.n = 50`) are sampled with a fixed seed from the papers that have full
text and an abstract.

- **ROUGE-1/2/L F1** (`rouge-score`, Porter stemming) between the abstract
  and the summary's tldr, problem, approach and findings. ROUGE counts
  shared words, so it is low for any abstractive summary and rewards
  copying; it is reported next to a *mismatched* baseline, each summary
  scored against another paper's abstract from the same corpus, which shows
  how much overlap the field's shared vocabulary produces on its own. Only
  the gap between the two columns says something about the summary.
- **Key-point recall.** A judge request (role `agents.judge`) extracts 3-6
  key points from the abstract: the problem, the approach and each main
  result, keeping the abstract's numbers. A second request shows the judge
  the key points and the summary, *not* the abstract, and asks which points
  the summary conveys. Recall is the share covered. The extraction depends
  only on the abstract, so it is made once per paper and served from the
  cache for every summary compared with it. A malformed judgment (a point
  left out or numbered twice) counts as not covered, so it can only lower
  recall.

Means come with 95% percentile-bootstrap intervals (2,000 resamples). With
50 papers the interval on recall is about +-5 points, which is the
resolution at which two configurations can be told apart.

**Caveats.**

- *The abstract is not ground truth.* Abstracts oversell, omit limitations
  and sometimes state results the body qualifies. A summary that is more
  careful than the abstract loses recall for it.
- *Judge bias.* LLM judges are lenient towards paraphrase and can be swayed
  by fluent, longer text. The judge sees one summary at a time with a
  binary question per key point, which limits but does not remove this.
- *Same-family judge.* By default the reader, the critic and the judge are
  all `claude-opus-5-5`. A model judging its own family's writing tends to
  rate it favourably (self-preference), so absolute recall is likely
  optimistic; comparisons between configurations judged by the same judge
  are more trustworthy than the absolute number. Setting
  `[agents.judge] model` to another model is the cheap check.

### 4.2 Faithfulness (`precis eval faithfulness`)

Computed from stored results, with no requests:

- **Verdict rates per claim** (supported, partial, unsupported,
  contradicted), where a claim is the tldr or one finding, for all first
  drafts, all final summaries, and the revised papers alone before and
  after revision.
- **Quote audit.** Every check must quote the paper verbatim. Each quote is
  searched for in the paper's text after folding case, Unicode forms,
  curly quotes and whitespace, with `...` treated as a gap. A quote that is
  not found means the critic paraphrased or invented its evidence; the
  audit reports how often, and how many of those checks still said
  "supported".
- **Examples**: failed claims from revised papers with the critic's note and
  the rewritten claim.

**Caveats.** The verdicts are a model's judgments, not ground truth; the
critic is the same model family as the reader. The revised-only
before/after comparison has a selection effect: those claims were chosen
because they failed, and the same critic re-checks text written in
response to its own notes, so the "after" rate is an upper bound on how
much revision helps. A random sample of verdicts checked by hand is the
honest calibration; section 6 has a place for it.

### 4.3 Strategy ablation (`precis eval ablation --n N`)

The three summarization strategies run on the same `N` papers, all without
the abstract: `agentic` (the reader navigates sections with tools),
`fulltext` (one request with the whole body) and `mapreduce` (notes per
chunk, then a combining request). Each summary is then scored with the same
critic and judge. Reported per strategy: requests, input, cached and output
tokens, and cost per paper; p50 and p95 latency per paper; key-point recall
and ROUGE-L; supported claims.

Tokens and cost come from the requests the run made; a request served from
the response cache is counted with the usage of the original response, so
a re-run reports the same cost. Latency is wall time per paper on the
gateway's clock and is reported only for papers whose requests were all
sent, not served from the cache.

## 5. Benchmarks

The benchmarks run the real per-paper pipeline (scheduler, agents, gateway,
store) against `FakeAnthropic`, which serves deterministic offline
responses with a latency model and enforces server-side rate limits with
`429` and `retry-after`, like the API.

**Virtual time.** The scaled clock from `precis.llm` runs simulated time a
fixed factor faster than wall time. That breaks down at this scale: 2,000
papers make about 25,000 requests, and at a 4,000 requests/minute limit any
useful speed-up asks the event loop for tens of thousands of requests per
wall second; it falls behind and its own CPU time turns into simulated
latency. `precis.eval.simclock.VirtualClock` instead advances time only
when every task is waiting on the clock, jumping straight to the next
wake-up. A simulated hour then costs only the CPU time of processing its
requests, the result does not depend on the machine, and the same seeds
give exactly the same run.

<!-- BENCH-METHOD -->

## 6. Results

<!-- BENCH-RESULTS -->

### Real-model results

> **To be filled in from a run against the API** (`precis eval abstract`,
> `precis eval faithfulness`, `precis eval ablation`). The commands write
> their tables to `results/*.md`.

| Measure | Value |
|---|---|
| Key-point recall, summaries without the abstract | *pending* |
| ROUGE-L F1, own abstract / mismatched abstract | *pending* |
| Supported claims, first drafts / final summaries | *pending* |
| Quotes not found verbatim | *pending* |
| Hand-checked verdicts that agree with the critic | *pending* |
| Ablation (agentic / fulltext / mapreduce) | *pending* |
