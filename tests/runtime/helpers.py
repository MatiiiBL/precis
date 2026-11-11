"""Request builders and responders shared by the runtime tests."""

from typing import Any

from precis.config import Config
from precis.llm.fake import FakeReply


def config(**sections: dict[str, Any]) -> Config:
    """A default config with some section fields overridden."""
    return Config.model_validate(sections)


def request(text: str = "Summarize the paper.", **extra: Any) -> dict[str, Any]:
    return {
        "model": "claude-opus-5-5",
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": text}],
        **extra,
    }


def echo(params: dict[str, Any]) -> FakeReply:
    """Reply with the last user message, so different requests get different replies."""
    content = params["messages"][-1]["content"]
    text = content if isinstance(content, str) else str(content)
    return FakeReply.text(f"echo: {text}")
