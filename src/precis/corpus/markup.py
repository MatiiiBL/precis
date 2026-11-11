"""Text helpers shared by the Markdown report and the HTML site."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

from markdown_it import MarkdownIt
from markupsafe import Markup

from precis.corpus.synthesis import CITATION, citation
from precis.models import PaperMeta

# CommonMark with raw HTML disabled: model-written text is rendered, never
# trusted, so a stray "<script>" shows up as text.
_MARKDOWN = MarkdownIt("commonmark", {"html": False})


def count(n: int, noun: str) -> str:
    """``count(1, "paper")`` -> "1 paper", ``count(2500, "paper")`` -> "2,500 papers"."""
    return f"{n:,} {noun}{'' if n == 1 else 's'}"


def slugify(text: str) -> str:
    """GitHub's heading anchor: lowercase, punctuation dropped, spaces to hyphens."""
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def page_name(paper_id: str) -> str:
    """A file name for a paper's page ("local:ab12" -> "local_ab12")."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)


def arxiv_links(meta: PaperMeta) -> dict[str, str]:
    """Abstract, PDF and HTML links for an arXiv paper; empty for local papers."""
    if meta.id.startswith("local:"):
        return {}
    versioned = f"{meta.id}v{meta.version}" if meta.version else meta.id
    return {
        "abs": f"https://arxiv.org/abs/{versioned}",
        "pdf": f"https://arxiv.org/pdf/{versioned}",
        "html": f"https://arxiv.org/html/{versioned}",
    }


def link_citations(text: str, href: Callable[[str], str | None]) -> str:
    """Turn ``[arXiv:<id>]`` tags into Markdown links where ``href`` gives a target."""

    def replace(match: re.Match[str]) -> str:
        lead = match.group(0)[: len(match.group(0)) - len(match.group(0).lstrip())]
        tags = []
        for part in re.split(r"[,;]", match.group(1)):
            paper_id = part.strip().removeprefix("arXiv:").strip()
            target = href(paper_id)
            tags.append(f"[arXiv:{paper_id}]({target})" if target else citation(paper_id))
        return lead + " ".join(tags)

    return CITATION.sub(replace, text)


# Both renderers escape raw HTML (see _MARKDOWN), so their output is safe to
# mark as such for Jinja2's autoescaping.


def markdown_html(text: str) -> Markup:
    """Render model-written Markdown to HTML."""
    return Markup(_MARKDOWN.render(text))


def inline_html(text: str) -> Markup:
    """Render one line of Markdown, without a surrounding paragraph."""
    return Markup(_MARKDOWN.renderInline(text))
