"""System prompts for every summarization role, kept as Markdown files.

A prompt's version is a hash of its final text (after includes), and the
stage ``config_hash`` contains it, so editing a prompt re-runs exactly the
stages that use it. Prompts are static: anything that varies per paper goes
into the first user message, which keeps the system prompt a cacheable prefix
shared by every paper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Literal

from precis.util import sha256_hex

PromptName = Literal[
    "reader", "critic", "reviser", "fulltext", "mapreduce_notes", "mapreduce_combine"
]

_INCLUDE = re.compile(r"<!-- include: (\w+) -->")


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str

    @property
    def version(self) -> str:
        """First 12 hex digits of the SHA-256 of the text."""
        return sha256_hex(self.text)[:12]


def _read(name: str) -> str:
    return files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")


@cache
def load_prompt(name: PromptName) -> Prompt:
    """Load a prompt, expanding ``<!-- include: other -->`` lines with ``other.md``."""
    text = _INCLUDE.sub(lambda m: _read(m.group(1)).strip(), _read(name))
    return Prompt(name=name, text=text.strip() + "\n")
