import pytest
from pydantic import BaseModel

from corpus.factories import ScriptedGateway, message
from precis.config import AgentConfig
from precis.corpus.structured import StructuredOutputError, request_structured, structured_params


class Answer(BaseModel):
    label: str
    items: list[str]


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
    assert params["output_config"]["effort"] == "low"
    schema = params["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["label", "items"]
    for forbidden in ("temperature", "top_p", "top_k", "thinking", "tool_choice"):
        assert forbidden not in params


async def test_reply_text_is_parsed() -> None:
    gateway = ScriptedGateway(lambda p, s: {"label": "x", "items": ["a"]})
    params = structured_params(AgentConfig(), system="s", user="u", output=Answer)
    answer = await request_structured(gateway, params, Answer, stage="label", paper_id="p1")
    assert answer == Answer(label="x", items=["a"])
    assert (gateway.calls[0].stage, gateway.calls[0].paper_id) == ("label", "p1")


async def test_truncated_reply_is_an_error() -> None:
    gateway = ScriptedGateway(lambda p, s: message('{"label": "x", "it', stop_reason="max_tokens"))
    params = structured_params(AgentConfig(), system="s", user="u", output=Answer)
    with pytest.raises(StructuredOutputError, match="max_tokens"):
        await request_structured(gateway, params, Answer, stage="label")


async def test_reply_of_the_wrong_shape_is_an_error() -> None:
    gateway = ScriptedGateway(lambda p, s: {"label": "x"})
    params = structured_params(AgentConfig(), system="s", user="u", output=Answer)
    with pytest.raises(StructuredOutputError, match="Answer"):
        await request_structured(gateway, params, Answer, stage="label")
