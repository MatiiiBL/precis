from typing import Any

import pytest
from pydantic import BaseModel

from corpus.factories import ScriptedGateway, message
from precis.agents.errors import AgentError, AgentRefused
from precis.agents.trace import AgentTrace
from precis.config import AgentConfig
from precis.corpus.structured import StructuredOutputError, request_structured, structured_params


class Answer(BaseModel):
    label: str
    items: list[str]


def _params() -> dict[str, Any]:
    return structured_params(AgentConfig(max_tokens=1000), system="s", user="u", output=Answer)


def test_params_follow_the_model_rules() -> None:
    params = structured_params(
        AgentConfig(model="claude-opus-5-5", effort="low", max_tokens=2000),
        system="sys",
        user="question",
        output=Answer,
    )
    assert params["model"] == "claude-opus-5-5"
    assert params["max_tokens"] == 2000
    assert params["messages"] == [{"role": "user", "content": "question"}]
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"]["effort"] == "low"
    schema = params["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["label", "items"]
    for forbidden in ("temperature", "top_p", "top_k", "tool_choice", "cache_control"):
        assert forbidden not in params


async def test_reply_text_is_parsed_and_usage_traced() -> None:
    gateway = ScriptedGateway(lambda p, s: {"label": "x", "items": ["a"]})
    trace = AgentTrace()
    answer = await request_structured(
        gateway, _params(), Answer, stage="label", paper_id="p1", trace=trace
    )
    assert answer == Answer(label="x", items=["a"])
    assert (gateway.calls[0].stage, gateway.calls[0].paper_id) == ("label", "p1")
    assert (trace.requests, trace.output_tokens) == (1, 50)


async def test_truncated_reply_is_retried_with_twice_the_limit() -> None:
    replies = iter([message('{"label": "x", "it', stop_reason="max_tokens"),
                    message('{"label": "x", "items": []}')])  # fmt: skip
    gateway = ScriptedGateway(lambda p, s: next(replies))
    answer = await request_structured(gateway, _params(), Answer, stage="label")
    assert answer.items == []
    assert [c.params["max_tokens"] for c in gateway.calls] == [1000, 2000]


async def test_truncated_twice_is_an_error() -> None:
    gateway = ScriptedGateway(lambda p, s: message("{", stop_reason="max_tokens"))
    with pytest.raises(AgentError, match="max_tokens"):
        await request_structured(gateway, _params(), Answer, stage="label")


async def test_refusal_raises() -> None:
    gateway = ScriptedGateway(lambda p, s: message("", stop_reason="refusal"))
    with pytest.raises(AgentRefused):
        await request_structured(gateway, _params(), Answer, stage="label")


async def test_reply_of_the_wrong_shape_is_an_error() -> None:
    gateway = ScriptedGateway(lambda p, s: {"label": "x"})
    with pytest.raises(StructuredOutputError, match="Answer"):
        await request_structured(gateway, _params(), Answer, stage="label")
    gateway = ScriptedGateway(lambda p, s: message("not json"))
    with pytest.raises(StructuredOutputError):
        await request_structured(gateway, _params(), Answer, stage="label")
