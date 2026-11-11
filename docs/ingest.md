# Ingest: from an arXiv query to documents an agent can read

The ingest package turns "papers in cs.AR submitted since July 2025" into
`PaperMeta` and `Document` records in the store, and gives the agents a
passage search over each document. This page explains how, why it is built
that way, and what it measured on the real corpus.

```
 precis fetch arxiv ─┐
 precis fetch ids  ──┼─▶ PaperMeta ─▶ HTML (LaTeXML) ──┬─▶ parse ─▶ quality gate ─▶ Document
 precis fetch local ─┘                 └─ 404 / placeholder / rejected ─▶ PDF ─▶ parse ─┘
                                                                     PaperIndex (BM25) ◀── agents
```

| Module | Job |
|---|---|
| `sources/http.py` | The one HTTP client for arXiv: shared pacing, User-Agent, retries |
| `sources/atom.py`, `sources/arxiv.py` | API queries, paging, date windows, Atom to `PaperMeta` |
| `sources/fulltext.py`, `sources/raw.py` | HTML-then-PDF download, raw file cache |
| `sources/local.py` | A directory of PDFs to papers |
| `sources/runner.py`, `cli/fetch.py` | The fetch stage against the store; `precis fetch` |
| `text/html.py`, `text/pdf.py` | LaTeXML HTML and PDF to sections and captions |
| `text/document.py`, `text/quality.py` | Section ids and the quality gate |
| `text/bm25.py` | `PaperIndex(doc).search(query, k)` |

## Measured on the real corpus

Measured on 2026-11-11 with `precis fetch arxiv --category cs.AR --since
2025-07-01 --metadata-only`, then `precis fetch ids` on a seeded random
sample of 50 of those papers, into a scratch store (commands at the end).

| | |
|---|---|
| Papers in cs.AR, 2025-07-01 to 2026-11-11 | **3,013** (2,031 with cs.AR as primary, 982 cross-listed: cs.LG 202, cs.CR 153, cs.DC 113, quant-ph 64, cs.ET 58, ...) |
| Papers with more than one version | 684 (23%) |
| Metadata search | 16 API requests, 45 s, no retries, no empty pages |
| Full text, 50 papers | **47 HTML (94%)**, 2 PDF fallbacks (4%), 1 failure (2%) |
| Why the 2 fell back | both: no HTML version (HTTP 404); both are PDF-only submissions made with Word |
| Why the 1 failed | `2512.24713v2` is a withdrawal notice: HTML and PDF both 404 |
| HTML pages rejected by the quality gate or as placeholders | 0 |
| Download time | 53 requests in 157 s: 3.0 s per request, the configured pace |
| Parse time per paper | HTML median 12 ms (max 60 ms); PDF median 383 ms (max 1.1 s) |
| Raw files on disk | HTML 49 kB per paper gzipped (6.2x smaller); PDFs 0.9 MB (gzip saves 10%) |

Document size, in `util.estimate_tokens` tokens (characters / 4), over the
49 parsed documents:

| min | p10 | p25 | median | p75 | p90 | max | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,894 | 4,783 | 6,093 | **8,283** | 12,763 | 16,216 | 26,506 | 9,705 |

21 of 49 documents are over 10k tokens and 2 over 20k; nothing is
truncated anywhere in ingest. The reader pages through a document instead of
loading it whole; how its token budget counts the conversation is in
[agents.md](agents.md). A document has a
median of 21 sections (max 56) and 11 captions (max 53); a section is a
median of 317 tokens (p90 955, max 2,909), which is the unit `read_section`
pages through. 10 of 49 papers have appendices.

Licenses of the 49 fetched papers: CC BY 4.0 23, arXiv's non-exclusive
license 18, CC BY-NC-ND 3, CC BY-NC-SA 2, CC BY-SA 1, unknown 2 (the
Word-made PDFs carry no license metadata).

**A full run** of the 3,013 papers takes about 1.06 requests per paper at
3 s each, so roughly 2.7 hours of polite downloading, and about 0.3 GB of
raw files (94% HTML at 49 kB, 6% PDF at 0.9 MB). Re-running it costs 16
search requests: every fetched paper is skipped from the store.

## Searching arXiv

A search is `(cat:cs.AR OR cat:cs.PF) AND submittedDate:[202507010000 TO
202611112359]`, sorted by submission date ascending and read in pages of
`fetch.page_size` (200). Three behaviours of the API shape the code:

- **Result cap.** Paging deep into a query is slow and drops pages, so a
  date range whose `totalResults` exceeds `RESULT_CAP` (10,000) is split in
  half, recursively down to one-minute windows, and each window is paged
  separately. The cs.AR range above is 3,013 results, one window.
- **Empty pages.** The API sometimes answers a valid request with zero
  entries. An empty page before the reported total is asked for again (up
  to `fetch.max_attempts`), and a window that reports zero results is
  checked once more before it is believed. The paging offset advances by
  the entries actually received, since pages can come back short.
- **Shifting pages.** A paper can move between pages while a long query
  is being read; results are de-duplicated by id.

Each Atom entry becomes a `PaperMeta`: id and version from the entry URL
(old-style ids such as `hep-th/9901001` included), every category and the
primary one, dates, and the license when the feed has one. The
current API never includes a license, so it is filled in from the HTML
page's info box ("License: CC BY 4.0", mapped to the canonical URL) or from
the license URL arXiv stamps into its PDFs. Re-running a search does not
erase a license learned this way.

## Being a polite client

All requests, to the API and to arxiv.org, go through one `ArxivHttp`:

- one `RateLimiter` starts requests at least `fetch.min_interval_s` (3 s)
  apart across every task, so four fetch workers still make one request
  every 3 s;
- every request carries `fetch.user_agent`. The default, `precis/<version>`,
  names the tool but nobody to contact; before a large run, add a contact
  so arXiv can reach you instead of blocking an unknown heavy client:

  ```toml
  # precis.toml
  [fetch]
  user_agent = "precis/0.1.0 (mailto:you@example.org)"
  ```

  The default deliberately carries no address, so nobody's is sent unless
  they configure it;
- 429, 5xx, timeouts and connection errors retry with exponential backoff
  and full jitter (a random wait up to 2 s, 4 s, 8 s, ..., at most 60 s), up to
  `fetch.max_attempts`. A `Retry-After` header (seconds or HTTP date,
  capped at 5 minutes) is honoured *for every caller*: it pushes back the
  shared limiter, not just the request that was refused.

The limiter takes the runtime's injectable `Clock`, so the tests check exact
spacing and backoff with `ManualClock` instead of sleeping.

## Full text: HTML first, PDF as the fallback

arXiv renders most LaTeX submissions to HTML with LaTeXML, and that HTML
keeps what a PDF loses: the section tree, math as LaTeX, footnotes in
place, table cells. So `FullTextFetcher` asks for
`https://arxiv.org/html/<id>v<n>` first and falls back to
`https://arxiv.org/pdf/<id>v<n>` when:

1. the HTML is a 404 (no HTML version: PDF-only submissions, failed
   conversions);
2. the page is not a LaTeXML document (a placeholder) or carries arXiv's
   "Conversion to HTML had a Fatal error" banner (the text may be cut off);
3. the HTML parses, but the quality gate rejects the result.

When neither format works, the error names every reason, e.g. `no PDF (HTTP
404; withdrawn versions have no full text); no HTML version (HTTP 404)`.

Raw files are kept gzipped under `fetch.raw_dir` (`html/2507.00367v1.html.gz`,
`pdf/...`), written atomically. A missing HTML version is remembered as a
`.missing` note with its reason, so a re-run, or a re-parse after a parser
change, makes no request for a paper it has seen; a refusal such as a 403
is not remembered, since it may not be permanent. Parsing runs in a worker
thread: a large PDF takes a second of CPU, and the event loop keeps
downloading meanwhile.

## The HTML parser

`text/html.py` walks the `ltx_document` article and ignores everything
outside it (arXiv's page chrome, table of contents, license box).

- **Sections.** Nested `<section class="ltx_section|ltx_subsection|...">`
  become `Section` records with positional ids: `s1`, `s1.2`, `s1.2.1`, and
  `a1`, `a1.1` for appendices. Positional ids match the paper's own
  numbering whenever the paper numbers sections normally, and stay stable
  and valid when it does not (unnumbered sections, a paper that starts at
  `\subsection`). A section's text is its own text only; its subsections
  are separate records, so section token counts add up to the document's.
  `\paragraph` blocks stay in their section with the run-in title in front.
- **Math** is rendered as `$<alttext>$`, the LaTeX source LaTeXML keeps in
  each `<math>`, with `\displaystyle` noise removed; a display equation
  keeps its number (`$E=mc^{2}$ (3)`), and each row of an aligned group is
  its own `$...$`.
- **Footnotes** stay where they are referenced, as `(Footnote: ...)`.
- **Floats** become `Caption` records (`fig1`, `tab2`, ...), with subfigure
  captions appended and table cells as `cell | cell` rows, and are left
  out of the section text. Algorithm floats count as figures.
- **Dropped:** title, authors, abstract, keywords, CCS classification,
  front-matter notes, acknowledgements (by class or by title),
  bibliography, and unconverted macro names (`ltx_ERROR`). An
  acknowledgment written as a bold lead-in instead of a section
  ("**Acknowledgment** This work is sponsored by ... Grant No. CCF2403134")
  is cut from the lead-in to the end of its paragraph, in both parsers;
  left in, its grant numbers read like results of the paper.
- A paper without any `\section` is cut into parts (see below).

The page is decoded as UTF-8 regardless of its declarations; without that,
lxml falls back to Latin-1 and turns every `•` into mojibake.

## The PDF parser

A PDF has positioned glyphs, not structure. `text/pdf.py` works on the
lines pypdf extracts (upright text only, which also drops the rotated arXiv
stamp in the margin) and rebuilds the structure with heuristics tuned on
IEEE and ACM two-column papers:

1. **Page furniture.** A line that recurs (digits ignored) in the top or
   bottom two lines of at least 30% of the pages is a running header or
   footer; a bare number there is a page number. ACM permission blocks,
   IEEE copyright lines and DOI lines are dropped. So is a **table of
   contents**: a run of entries ending in a page number with no running
   text between them, at least three of them numbered or with dot leaders
   and titled with real words (so table rows such as
   `3.1 56 X 56 X 128 3 X 3 X 128 X 256` never qualify).
2. **References.** Everything from the last `References`/`Bibliography`
   heading (with or without a trailing period or colon) is dropped, except
   an appendix after it: either a line that says
   "Appendix", or lettered headings (`A Allocation Policies`, `A.1 ...`)
   counting from A after the last numbered reference. Lines that look like
   author lists (initials, years, "et al") are never appendix headings, and
   neither are sentences that start with the article ("A Hopfield neural
   network is ..."): a lettered heading is at most 90 characters and does
   not run on into a lowercase line.
3. **Headings** are found by pattern: `2 Background`, `2.3 Title`,
   `II. BACKGROUND` (with small caps repaired: `I NTRODUCTION`),
   `A. Subsection`, IEEE run-ins `1) Title: text`, ACM run-ins
   `4.1.1 Memory Requirements. Working set ...` (title split from the
   text), and a number alone on its line above its title. A title that
   wrapped is joined with its second line when it broke inside a word, on
   a comma or connector ("... Architecture with" / "Blackwell Instances"),
   or when a Title Case heading that filled its line is followed by a short
   Title Case fragment and then running text ("... Signal Processing and
   Deep" / "Learning"); a short heading followed by a capitalized line is
   left alone, since that line is a sub-heading ("FFT Engine" / "Theory of
   Operation"). Titles may be up to 120 characters.
   Patterns alone match figure labels, axis ticks and numbered lists, so
   the parser then keeps the **best consistently numbered chain**: a small
   dynamic program over the candidates picks the in-order sequence 1, 2, 3,
   ... with the most weight (Title Case and ALL CAPS titles weigh more than
   sentences, so "1. We propose ..." loses to "1 Introduction"; a candidate
   whose next line continues it in lowercase, like "... shown in Table" /
   "3. ICE models ..., validated across" / "multiple platforms", weighs
   half; one missed number is tolerated at half credit), then does the same
   for each section's subsections. Papers without numbered headings fall
   back to standalone all-caps lines and common section names followed by
   prose.
   **Lopsided outlines are distrusted.** If fewer than 6 sections come out
   and one holds over 80% of a paper of at least 3,000 tokens, the headings
   were not the paper's (a table of contents, list items) and most were
   missed. The parser retries with unnumbered headings and, if that is no
   better, cuts the paper into parts (see below): no outline beats a wrong
   one, since the reader trusts the titles it is shown.
4. **Front matter** (title, authors, abstract, keywords) is everything
   before the first heading and is dropped; Abstract, Acknowledgments and
   Keywords headings anywhere cut their text out, and so does a run-in
   acknowledgment (see the HTML parser).
5. **Captions** (`Fig. 3: ...`, `TABLE II` + title line) are moved into
   `Caption` records. Without font information a caption's end is a guess,
   so it stops at the first sentence end or short line: a long caption's
   tail can land in the body, but body text never lands in a caption.
6. **Debris.** Runs of three or more short non-prose lines (figure labels,
   table cells, equations mangled by extraction) are removed. A line is
   prose when it has at least three all-lowercase words.
7. **Paragraphs and hyphenation.** Lines join into paragraphs (a short line
   ending a sentence ends a paragraph; bullets start one). A word
   hyphenated across a line break is rejoined using the document's own
   vocabulary: `archi-`/`tecture` becomes "architecture" because that word
   occurs elsewhere, `column-`/`major` stays "column-major" because the
   hyphenated form does, and compounds with acronyms or numbers
   (`CKKS-friendly`, `64-bit`) keep their hyphen. Word's spaced hyphens
   ("safety -critical") are closed up.

**How well it works.** Eight of the sample papers were also downloaded as
PDFs, so the PDF parse could be compared with the HTML parse of the same
paper (the HTML structure serves as ground truth). Over those eight, the PDF
parser recovers **214 of 224 section titles (96%)**, with three spurious
ones (two are real headings with mangled math, one an "Appendix" wrapper),
and between 0.97 and 1.31 times the HTML's tokens (median 1.02). On the
IEEE fixture the outline, ids included, is identical to the HTML's. The
first version of the parser recovered 85% of the top two levels of titles
on the same papers; the run-in, wrapped-title and IEEE `1)` handling closed
most of the gap, and lettered appendices after author-year references took
one paper from 42% of its HTML tokens to 108%.

**Known limitations of the PDF path** (4% of the sample uses it):
pypdf drops the space where small caps change size ("RELATEDWORK"), and
Word exports split words ("efficac y"); equations come out as linear text
or are dropped as debris; subfigure captions and algorithm listings stay in
the body (the source of the 1.31 ratio above); two-column order relies on
the PDF's content order, which LaTeX gets right and some tools do not; and
a scanned PDF has no text and fails with that reason (no OCR).

## The quality gate

A document that is too short or garbled would still be a valid `Document`,
and an agent would then write a confident summary of nothing. So the fetch
stage fails it instead, with the reason in the stage record:

| Check | Threshold | Real documents (57, HTML and PDF) |
|---|---|---|
| Body text | at least 500 tokens | min 2,894 |
| Control, private-use or replacement characters | at most 5% | max 0.03% |
| Tokens longer than 25 characters (lost spacing) | at most 5% | max 1.3% |
| Common English function words (the, of, and, ...) | at least 10% | min 18.3%, median 28% |

Failed font decoding produces either junk code points or letters that form
no words; both are far outside the range of real papers. The function-word
check assumes English, which holds for cs.AR. An HTML page that fails the
gate falls back to the PDF before the paper is failed.

## Documents without headings

A document without detectable headings is not one giant section: it is cut
into sections titled "Part 1", "Part 2", ... of about 1,500 tokens, between
paragraphs where possible, then between sentences, and inside a sentence
only when one sentence alone is longer than a part. Nothing is dropped;
a property test checks that every word survives in order.

## Passage search (`PaperIndex`)

`PaperIndex(doc)` cuts every section into passages of about 120 words at
sentence boundaries, never across sections, so every hit has a section id
the agent can then read in full. A sentence over 180 words is cut by word
count, and a short tail joins the passage before it. Passage ids are
`<section_id>#p<n>`.

Ranking is Okapi BM25 (k1 = 1.5, b = 0.75, idf = ln(1 + (N - df + 0.5) /
(df + 0.5))). The tokenizer lowercases, splits on non-alphanumerics
(`KV-caches` gives `kv`, `cache`), drops 33 stopwords and folds
plurals crudely (`GPUs` and `GPU` match; queries and passages are folded
the same way, so `analysis` becoming `analysi` is harmless). Over the
sample, a document has a median of 42 passages (HTML) and building the
index plus one search takes 2.7 ms median, 8.8 ms max. Captions are not
indexed (they have no section); the agents read them with
`read_captions`.

## Local PDFs

`precis fetch local <dir>` reads every `*.pdf` below a directory. The id is
`local:` plus the first 16 hex digits of the file's SHA-256, so moving or
renaming a file keeps its identity and an identical copy is the same paper.
The metadata arXiv would provide comes from the PDF: title and authors from
its document information (the title falls back to the largest text on page
one), the abstract from the front matter, the date from the creation date
or the file's modification time, category `local`.

## The fetch stage and the CLI

```
precis fetch arxiv --category cs.AR [--category ...] --since 2025-07-01 [--until ...] [--limit N] [--metadata-only]
precis fetch ids 2406.01234 2406.05678v2 ... [--metadata-only]
precis fetch local <dir> [--limit N]
```

Each command first stores the metadata (search results as they arrive),
then fetches full text for every paper whose `fetch` stage is not done. A
paper counts as done when its stage is `done`, its `config_hash` matches
(the parser version and `fetch.prefer`), its arXiv version is the one
fetched, and its document is stored. So:

- re-running a command resumes: done papers are skipped without a request,
  failed ones are tried again (`attempts` counts up);
- a new arXiv version is fetched again;
- bumping `runner.PARSER_VERSION` re-parses everything from the raw files,
  without downloading.

Each paper is recorded as it finishes: `running`, then `done` with a
summary in `StageRecord.output` (`version`, `source_format`, `n_tokens`,
`n_sections`, `n_captions`, `fallback_reason`) or `failed` with the reason.
The document is written before the stage, so a crash between the two
leaves `running`. Every fetch command starts with
`Store.reset_running(stages=[FETCH])`, which turns such rows back into
`pending` and says how many; it touches only `fetch` rows, so a `precis
run` working on the same store keeps its own.
Ctrl-C puts in-flight papers back to `pending` and exits with code 130; a
search that fails exits with code 1; individual paper failures do not
change the exit code (withdrawn papers fail on every run). Any exception
while fetching, parsing or storing one paper, expected or not, becomes that
paper's `failed` record with the error text; it never ends the command.
`--metadata-only`
marks new papers `pending`. Four workers run at once so parsing overlaps
downloading; the shared limiter keeps the request rate at one per 3 s.
Every invocation is recorded with `Store.start_run`/`finish_run` and its
counts. A rich progress view shows papers found, papers fetched (HTML, PDF,
failed, skipped) and an ETA, and a summary table follows.

Configuration (`[fetch]` in `precis.toml`):

| Key | Default | |
|---|---|---|
| `min_interval_s` | 3.0 | gap between request starts |
| `user_agent` | `precis/<version>` | sent with every request; add a contact (see above) |
| `prefer` | `"html"` | `"pdf"` skips the HTML |
| `raw_dir` | `data/raw` | downloaded files |
| `page_size` | 200 | results per API request |
| `max_attempts` | 5 | per request, and per empty page |
| `timeout_s` | 60 | per request |

## The full corpus

`precis fetch arxiv --category cs.AR --since 2025-07-01 --until 2026-10-31`,
run on 2026-11-11. The store and the raw files stay out of the repository;
the counts below come from the store's `fetch` stage records.

| | |
|---|---|
| Papers found | **2,918** |
| Full text from HTML | **2,670 (91.5%)** |
| Full text from PDF | 226 (7.7%): 211 have no HTML version; 15 HTML pages failed the quality gate (11 with words run together, 4 with no body text) |
| Failed | 22 (0.8%): 10 withdrawn (no PDF and no HTML), 7 PDFs whose text decodes to garbage (2-4% common English function words), 3 with under 500 tokens of body text, 2 PDFs with corrupted characters or lost word spacing |
| Download | 2,272 requests in 1 h 55 min for the run that finished it (3.0 s each), after a first run that stopped at 820 papers (the surrogate crash below) |
| Raw files | 545 MB: HTML 125 MB (gzipped), 238 PDFs 420 MB |

Document size in tokens, over the 2,896 parsed papers:

| min | p10 | p25 | median | p75 | p90 | p99 | max | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 798 | 3,871 | 5,885 | **8,670** | 13,021 | 16,253 | 37,323 | 129,035 | 10,131 |

29.3 million tokens in all; 132 documents are over 20,000 tokens and 19
over 40,000 (theses and long surveys). A document has a median of 21
sections and 12 captions. The 50-paper sample above predicted the shape
well: a median of 8,283 tokens and 94% HTML against 8,670 and 91.5%.

## After the full run: parser version 2

A review of the first papers of the full cs.AR run found four parsing
faults, all fixed in parser version 2, and the run itself hit one crash:

| Paper | Fault | Now |
|---|---|---|
| 2507.07223 (PDF) | the table of contents became the outline: 4 sections titled with page numbers, "Conclusion 51" holding 99.5% of the text | the 22 sections and subsections of the paper |
| 2507.12904 (PDF) | "Applications of CGRAs in Signal Processing and Deep" | "... and Deep Learning" |
| 2508.13249 (PDF) | "References." went unrecognized and numbered references became sections 7-12 | 6 sections, no references |
| 2507.10912 (HTML) | a bold "Acknowledgment" lead-in kept, grant numbers included | removed |
| 2512.14151 (PDF) | pypdf returned math letters as lone UTF-16 surrogates (`\ud835`); the store could not encode the document, and the error ended the whole fetch after 820 papers | surrogate pairs are recombined and lone halves dropped for all text; any error with one paper is recorded as that paper's failure and the run goes on |

Bumping the version re-parses every fetched paper from its raw file the
next time a fetch command runs; nothing is downloaded again (missing HTML
is remembered, and cached HTML that was rejected falls back to the cached
PDF). When version 2 was made, the running store had 721 papers fetched
(648 HTML, 73 PDF). Re-parsing all of them took about a minute and
changed 23 documents; the other 698 came out identical (over the whole
corpus, once the fetch had finished, re-parsing all 2,918 papers took 3.5
minutes and 25 requests to arXiv, and changed 92 of the 2,896 documents:
59 PDF, 33 HTML; 51 of the changes are PDF outlines):

- 14 PDF outlines: the table of contents above; wrapped titles completed in
  six papers; a missed subsection found (2509.04173); the reference list
  above; three lopsided outlines caught by the safety net, one rebuilt
  from unnumbered headings (2511.05502) and two cut into parts (2508.00295,
  a letter whose only heading is "Introduction", and 2510.26492); and two
  papers whose outlines were already made of numbered list items gain one
  more such item, now that titles may be longer (2510.07116, 2511.11917);
- 3 PDF texts: two run-in acknowledgments removed (in the two-page
  2508.17562 together with the reference list that followed it in the same
  paragraph), and one reference list under "References:" dropped;
- 6 HTML texts: a run-in acknowledgment removed, nothing else.

## Tests

`tests/ingest/` covers every module without the network: the HTTP layer
against `httpx.MockTransport` with a `ManualClock`; search against a fake
API that pages, splits windows and injects empty pages; downloads against a
fake arxiv.org; the CLI end to end through Typer's runner on a real store.
Parser tests use synthetic LaTeXML pages and PDFs generated in the test
(`helpers.make_pdf` writes a minimal text PDF), plus two CC BY 4.0 papers
in `tests/fixtures/papers/` (see its README): an IEEE paper in both formats
and a preprint with lettered appendices after author-year references.
`pytest -m network` runs one smoke test against the real arXiv.

## Reproducing the measurements

```sh
cat > measure.toml <<'EOF'
store_path = "scratch/measure.db"
[fetch]
raw_dir = "scratch/raw"
EOF
uv run precis --config measure.toml fetch arxiv --category cs.AR \
    --since 2025-07-01 --until 2026-11-11 --metadata-only
uv run precis --config measure.toml fetch ids <50 ids>   # the sample
```

The sample was `random.Random(2025).sample(papers, 50)` over the search
results in submission order.
