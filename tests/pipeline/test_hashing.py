import pytest

from precis.agents import messages
from precis.config import AgentConfig, AgentsConfig
from precis.models import ClaimCheck, PaperSummary, Verdict, Verification
from precis.pipeline.hashing import read_hash, revise_hash, verify_hash

from .helpers import paper


def test_every_stage_hash_covers_the_request_format(monkeypatch: pytest.MonkeyPatch) -> None:
    meta, doc = paper(0)
    config = AgentsConfig()
    check = ClaimCheck(target="tldr", verdict=Verdict.UNSUPPORTED, evidence="", note="n")
    verification = Verification(paper_id=meta.id, revision=0, checks=[check])

    def hashes() -> list[str]:
        return [
            read_hash(config, meta, doc),
            verify_hash(config, meta, doc, summary()),
            revise_hash(config, meta, doc, summary(), verification),
        ]

    before = hashes()
    monkeypatch.setattr(messages, "REQUEST_FORMAT_VERSION", messages.REQUEST_FORMAT_VERSION + 1)
    after = hashes()
    assert all(b != a for b, a in zip(before, after, strict=True))


def summary(tldr: str = "Design0 is fast.") -> PaperSummary:
    return PaperSummary(
        paper_id="2501.00000",
        tldr=tldr,
        problem="p",
        approach="a",
        findings=[],
        contributions=[],
        limitations=[],
        methods=[],
        datasets=[],
        keywords=[],
    )


def test_read_hash_tracks_reader_settings_strategy_and_input() -> None:
    meta, doc = paper(0)
    base = AgentsConfig()
    h = read_hash(base, meta, doc)
    assert h == read_hash(AgentsConfig(), meta, doc)
    assert h != read_hash(base.model_copy(update={"reader": AgentConfig(effort="high")}), meta, doc)
    assert h != read_hash(base.model_copy(update={"strategy": "fulltext"}), meta, doc)
    assert h != read_hash(base.model_copy(update={"include_abstract": False}), meta, doc)
    assert h != read_hash(base.model_copy(update={"page_tokens": 1000}), meta, doc)
    assert h != read_hash(base, meta, paper(1)[1])
    edited = meta.model_copy(update={"abstract": "Something else."})
    assert h != read_hash(base, edited, doc)
    # Settings of other roles do not matter to the reader.
    assert h == read_hash(base.model_copy(update={"critic": AgentConfig(effort="high")}), meta, doc)


def test_verify_and_revise_hashes_track_their_inputs() -> None:
    meta, doc = paper(0)
    config = AgentsConfig()
    h = verify_hash(config, meta, doc, summary())
    assert h != verify_hash(config, meta, doc, summary("Design0 is slow."))
    assert h != verify_hash(
        config.model_copy(update={"critic": AgentConfig(model="claude-haiku-5-5")}),
        meta,
        doc,
        summary(),
    )
    check = ClaimCheck(target="tldr", verdict=Verdict.UNSUPPORTED, evidence="", note="n")
    verification = Verification(paper_id=meta.id, revision=0, checks=[check])
    r = revise_hash(config, meta, doc, summary(), verification)
    assert r != revise_hash(
        config.model_copy(update={"revise": False}), meta, doc, summary(), verification
    )
    assert r != revise_hash(
        config.model_copy(update={"critic": AgentConfig(effort="low")}),
        meta,
        doc,
        summary(),
        verification,
    )
