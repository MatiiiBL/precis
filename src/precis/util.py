"""Small helpers shared across packages."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

# Claude's tokenizer averages a little under four characters per token on
# English prose with math and code mixed in. The estimate only sizes rate-limit
# reservations and paging; real usage always comes from the API response.
CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys and no whitespace, so equal objects hash equally."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()
