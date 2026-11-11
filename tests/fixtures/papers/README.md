# Paper fixtures

Real arXiv papers used to test the parsers on the markup and layouts the
pipeline meets in practice. Each is distributed by its authors under the
[Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
which permits redistribution and adaptation with attribution. The files here
are adapted as described below; the text itself is unchanged.

| File | Paper | Authors | Source |
|---|---|---|---|
| `2508.21267v1.html.gz` | *Catwalk: Unary Top-K for Efficient Ramp-No-Leak Neuron Design for Temporal Neural Networks* (arXiv:2508.21267v1) | Devon Lister, Prabhu Vellaisamy, John Paul Shen, Di Wu | <https://arxiv.org/html/2508.21267v1> |
| `2508.21267v1.pdf` | same paper | same | <https://arxiv.org/pdf/2508.21267v1> |
| `2610.02233v2-p1-13.pdf` | *Budgeted Cache Repair for Cross-Context KV-Cache Reuse* (arXiv:2610.02233v2) | Haeyong Kang, Chang D. Yoo | <https://arxiv.org/pdf/2610.02233v2> |

License: CC BY 4.0 for all three (the arXiv page of each paper shows the
license; the PDFs carry it in their document information as
`http://creativecommons.org/licenses/by/4.0/`).

## Changes made

- `2508.21267v1.html.gz`: arXiv's page chrome (scripts, styles, navigation,
  table of contents, footer) removed, keeping the `<article>` produced by
  LaTeXML and the license box; gzip-compressed.
- `2508.21267v1.pdf`: raster images and embedded font programs removed to
  shrink the file. Text, layout and fonts' character mappings are untouched,
  so text extraction gives exactly the same result as on the original.
- `2610.02233v2-p1-13.pdf`: pages 1-13 of 25 (body, references and the first
  two appendices), with images and font programs removed as above.

Why these two: the first is a two-column IEEE-style paper with roman-numbered
sections and small-caps headings, available as both HTML and PDF, so the PDF
parser can be checked against the HTML structure. The second is a
single-column preprint with author-year references followed by lettered
appendices (A, B, ...), the case where a parser that stops at "References"
loses a third of the paper.
