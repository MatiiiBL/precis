"""Downloaded files kept on disk, so parsing can be repeated without downloading again.

Files are gzip-compressed (LaTeXML HTML shrinks about 5x, PDFs a little) and
written atomically, so an interrupted run never leaves a truncated file that
a later run would trust. A paper whose HTML does not exist gets a small
``.missing`` note instead, so a resumed run goes straight to the PDF.
"""

from __future__ import annotations

import asyncio
import gzip
import os
from pathlib import Path
from typing import Literal

Kind = Literal["html", "pdf"]


class RawStore:
    """``<root>/<kind>/<id>v<version>.<kind>.gz``, one file per paper version and format."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, paper_id: str, version: int | None, kind: Kind) -> Path:
        name = paper_id.replace("/", "_") + (f"v{version}" if version is not None else "")
        return self.root / kind / f"{name}.{kind}.gz"

    async def load(self, paper_id: str, version: int | None, kind: Kind) -> bytes | None:
        path = self.path(paper_id, version, kind)
        return await asyncio.to_thread(_read, path)

    async def save(self, paper_id: str, version: int | None, kind: Kind, data: bytes) -> None:
        path = self.path(paper_id, version, kind)
        await asyncio.to_thread(_write, path, gzip.compress(data, mtime=0))

    async def mark_missing(
        self, paper_id: str, version: int | None, kind: Kind, reason: str
    ) -> None:
        """Remember that this format does not exist for the paper, and why."""
        path = self._missing_path(paper_id, version, kind)
        await asyncio.to_thread(_write, path, reason.encode())

    async def missing_reason(self, paper_id: str, version: int | None, kind: Kind) -> str | None:
        data = await asyncio.to_thread(_read_plain, self._missing_path(paper_id, version, kind))
        return None if data is None else data.decode()

    def _missing_path(self, paper_id: str, version: int | None, kind: Kind) -> Path:
        return self.path(paper_id, version, kind).with_suffix(".missing")


def _read(path: Path) -> bytes | None:
    data = _read_plain(path)
    return None if data is None else gzip.decompress(data)


def _read_plain(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
