from corpus.factories import make_meta, make_summary, make_verification, make_view
from precis.corpus.load import FinishedPaper
from precis.eval.faithfulness import (
    audit_quotes,
    evaluate_faithfulness,
    faithfulness_markdown,
    quote_found,
    revision_examples,
)
from precis.models import Caption, Document, Section, Verdict

S, P, U, C = Verdict.SUPPORTED, Verdict.PARTIAL, Verdict.UNSUPPORTED, Verdict.CONTRADICTED

TEXT = (
    "Our learned policy improves the hit rate by 7.2%\nover LRU on SPEC CPU2017. "
    "It costs “only” 3 KB of state."
)


def _doc(paper_id: str) -> Document:
    return Document(
        paper_id=paper_id,
        source_format="html",
        sections=[Section(id="s4", title="Results", level=1, text=TEXT, n_tokens=30)],
        captions=[Caption(id="tab1", kind="table", text="Table 1: hit rates per benchmark.")],
        n_tokens=30,
        sha256="0" * 64,
    )


def test_quotes_match_despite_wrapping_case_and_typography() -> None:
    text = " ".join(TEXT.casefold().split()).replace("“", '"').replace("”", '"')
    assert quote_found("improves the hit rate by 7.2% over LRU", text)
    assert quote_found('"It costs "only" 3 KB of state."', text)
    assert quote_found("Our learned policy ... over LRU on SPEC CPU2017", text)
    assert not quote_found("improves the hit rate by 9%", text)
    assert not quote_found("...", text)


def _view_with_evidence(
    paper_id: str, evidence: list[str], verdicts: list[Verdict]
) -> FinishedPaper:
    summary = make_summary(paper_id, ["caches", "prefetching"])
    check = make_verification(summary, verdicts)
    checks = [
        c.model_copy(update={"evidence": e}) for c, e in zip(check.checks, evidence, strict=True)
    ]
    check = check.model_copy(update={"checks": checks})
    return FinishedPaper(make_meta(paper_id), summary, check, summary, check)


def test_audit_counts_missing_and_empty_quotes() -> None:
    view = _view_with_evidence("p1", ["hit rate by 7.2%", "made-up sentence", ""], [S, S, U])
    audit = audit_quotes([view, make_view("p2", ["x", "y"])], {"p1": _doc("p1")})
    assert (audit.checks, audit.empty, audit.not_found, audit.not_found_supported) == (3, 1, 1, 1)
    assert audit.not_found_rate == 0.5


def test_revision_examples_pair_failed_claims_with_rewrites() -> None:
    papers = [
        make_view("b", ["x", "y"], draft_verdicts=[S, S, C], revised=True),
        make_view("a", ["x", "y"], draft_verdicts=[U, S, S], revised=True),
        make_view("c", ["x", "y"], draft_verdicts=[S, P, S]),
    ]
    examples = revision_examples(papers)
    assert [(e.paper_id, e.target, e.verdict) for e in examples] == [
        ("a", "tldr", U),
        ("b", "finding:1", C),
    ]
    assert examples[1].claim == papers[0].draft.findings[1].claim
    assert examples[1].revised_verdict is S
    assert revision_examples(papers, limit=1)[0].paper_id == "a"


def test_faithfulness_report() -> None:
    papers = [
        make_view("a", ["x", "y"], draft_verdicts=[U, S, S], revised=True),
        make_view("b", ["x", "y"], draft_verdicts=[S, P, S]),
    ]
    result = evaluate_faithfulness(papers, {"a": _doc("a")})
    assert (result.papers, result.flagged, result.revised) == (2, 1, 1)
    assert result.draft.unsupported == 1
    assert result.final.unsupported == 0
    text = faithfulness_markdown(result)
    assert "| All first drafts | 6 | 66.7% | 16.7% | 16.7% | 0.0% |" in text
    assert "| Revised papers, after | 3 | 100.0% | 0.0% | 0.0% | 0.0% |" in text
    assert "## Examples of revised claims" in text
