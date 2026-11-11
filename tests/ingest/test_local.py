import os
from datetime import date
from pathlib import Path

import pytest

from precis.sources.local import LOCAL_CATEGORY, find_pdfs, load_pdf, local_id
from precis.text.document import ParseError
from precis.text.quality import QualityError

from .helpers import FIXTURES, make_pdf, paper_pdf


def test_find_pdfs_recurses_and_ignores_other_files(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    for name in ("b.pdf", "A.PDF", "nested/c.pdf", "notes.txt", "d.pdf.bak"):
        (tmp_path / name).write_bytes(b"%PDF-1.4")
    assert [p.relative_to(tmp_path).as_posix() for p in find_pdfs(tmp_path)] == [
        "A.PDF",
        "b.pdf",
        "nested/c.pdf",
    ]
    with pytest.raises(NotADirectoryError):
        find_pdfs(tmp_path / "missing")


def test_a_local_pdf_becomes_a_paper(tmp_path: Path) -> None:
    path = tmp_path / "learned_caches.pdf"
    path.write_bytes(paper_pdf())
    paper = load_pdf(path)
    meta = paper.meta
    assert meta.id == local_id(path.read_bytes())
    assert meta.id.startswith("local:")
    assert len(meta.id) == len("local:") + 16
    assert paper.document.paper_id == meta.id
    assert meta.title == "A Synthetic Paper"  # largest text on page 1
    assert meta.abstract == "An abstract."
    assert (meta.categories, meta.primary_category) == ([LOCAL_CATEGORY], LOCAL_CATEGORY)
    assert meta.license == "http://creativecommons.org/licenses/by/4.0/"
    assert meta.url == str(path.resolve())
    assert [s.title for s in paper.document.sections] == [
        "Introduction",
        "Design",
        "Evaluation",
        "Conclusion",
    ]


def test_the_id_follows_the_content_not_the_name(tmp_path: Path) -> None:
    data = paper_pdf()
    (tmp_path / "one.pdf").write_bytes(data)
    (tmp_path / "copy of one.pdf").write_bytes(data)
    (tmp_path / "other.pdf").write_bytes(paper_pdf("OTHER"))
    ids = [load_pdf(p).meta.id for p in find_pdfs(tmp_path)]
    assert ids[0] == ids[1] != ids[2]


def test_dates_come_from_the_pdf_or_the_file(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(paper_pdf())
    os.utime(path, (1_700_000_000, 1_700_000_000))
    assert load_pdf(path).meta.published == date.fromtimestamp(1_700_000_000)
    dated = tmp_path / "dated.pdf"
    dated.write_bytes(paper_pdf(CreationDate="D:20240229090000Z"))
    assert load_pdf(dated).meta.published == date(2024, 2, 29)


def test_a_real_preprint() -> None:
    meta = load_pdf(FIXTURES / "2610.02233v2-p1-13.pdf").meta
    assert meta.title == "Budgeted Cache Repair for Cross-Context KV-Cache Reuse"
    assert meta.authors == ["Haeyong Kang", "Chang D. Yoo"]
    assert meta.abstract.startswith("Cross-context KV-cache reuse predicts")


def test_unusable_pdfs_raise(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 truncated")
    with pytest.raises(ParseError):
        load_pdf(broken)
    stub = tmp_path / "stub.pdf"
    stub.write_bytes(make_pdf([["1 Introduction", "Only a line of text."]]))
    with pytest.raises(QualityError):
        load_pdf(stub)
