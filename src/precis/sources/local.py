"""Papers from a local directory of PDFs, for work that is not on arXiv.

Each PDF becomes a ``PaperMeta`` and a ``Document``. The id is ``local:``
plus the first 16 hex digits of the file's SHA-256, so a file keeps its id
wherever it is moved or however it is renamed, and an identical copy is the
same paper. What arXiv metadata would provide comes from the PDF itself:
title and authors from its document information (or, for the title, the
largest text on the first page), the abstract from the front matter, and the
date from the creation date or, failing that, the file's modification time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from precis.models import Document, PaperMeta
from precis.text.pdf import read_pdf
from precis.text.quality import check_document
from precis.util import sha256_hex

LOCAL_CATEGORY = "local"


@dataclass(frozen=True)
class LocalPaper:
    meta: PaperMeta
    document: Document


def find_pdfs(directory: Path) -> list[Path]:
    """Every PDF under ``directory`` (any letter case, any depth), in a stable order."""
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def local_id(data: bytes) -> str:
    return f"local:{sha256_hex(data)[:16]}"


def load_pdf(path: Path) -> LocalPaper:
    """Read and parse one PDF. Blocking: call it from a worker thread.

    Raises ``OSError`` if the file cannot be read and ``ParseError`` (or its
    ``QualityError`` subclass) if it cannot be turned into a usable document.
    """
    data = path.read_bytes()
    paper_id = local_id(data)
    document, front = read_pdf(data, paper_id)
    check_document(document)
    meta = PaperMeta(
        id=paper_id,
        title=front.title or path.stem.replace("_", " "),
        authors=front.authors,
        abstract=front.abstract,
        categories=[LOCAL_CATEGORY],
        primary_category=LOCAL_CATEGORY,
        published=front.created or date.fromtimestamp(path.stat().st_mtime),
        license=front.license,
        url=str(path.resolve()),
    )
    return LocalPaper(meta=meta, document=document)
