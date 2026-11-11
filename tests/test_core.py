from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from precis.cli import app
from precis.config import Config, load_config
from precis.models import (
    ClaimCheck,
    Finding,
    PaperMeta,
    PaperSummary,
    Verdict,
    Verification,
)
from precis.util import canonical_json, estimate_tokens, sha256_hex


def _summary() -> PaperSummary:
    return PaperSummary(
        paper_id="2406.01234",
        tldr="A cache that learns.",
        problem="Caches miss.",
        approach="Learn the policy.",
        findings=[Finding(claim="Hit rate +7%.", section_ids=["s4"])],
        contributions=["A learned policy."],
        limitations=["Simulation only."],
        methods=["reinforcement learning"],
        datasets=["SPEC CPU2017"],
        keywords=["caches"],
    )


def test_records_round_trip_through_json() -> None:
    summary = _summary()
    assert PaperSummary.model_validate_json(summary.model_dump_json()) == summary


def test_records_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PaperMeta.model_validate(
            {
                "id": "x",
                "title": "t",
                "authors": [],
                "abstract": "",
                "categories": [],
                "primary_category": "cs.AR",
                "published": date(2026, 1, 1),
                "url": "u",
                "surprise": 1,
            }
        )


def test_verification_flags_unsupported_claims() -> None:
    def check(verdict: Verdict) -> ClaimCheck:
        return ClaimCheck(target="tldr", verdict=verdict, evidence="", note="")

    ok = Verification(paper_id="p", revision=0, checks=[check(Verdict.SUPPORTED)])
    bad = Verification(
        paper_id="p", revision=0, checks=[check(Verdict.SUPPORTED), check(Verdict.CONTRADICTED)]
    )
    assert not ok.needs_revision()
    assert bad.needs_revision()
    assert bad.supported_fraction() == 0.5


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})
    assert sha256_hex("abc") == sha256_hex(b"abc")


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd" * 10) == 10


def test_config_defaults_and_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_config() == Config()
    path = tmp_path / "custom.toml"
    path.write_text('[agents.reader]\nmodel = "claude-haiku-5-5"\n[budget]\nmax_cost_usd = 5\n')
    cfg = load_config(path)
    assert cfg.agents.reader.model == "claude-haiku-5-5"
    assert cfg.agents.critic.model == Config().agents.critic.model
    assert cfg.budget.max_cost_usd == 5


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("precis ")
