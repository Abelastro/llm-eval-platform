"""Tests for run comparison."""

from __future__ import annotations

from llm_eval.models import MetricResult, Run
from llm_eval.runners.compare import compare_runs


def _run(run_id: str, scores: dict[int, tuple[float, bool]]) -> Run:
    results = {
        index: {
            "bleu": MetricResult(name="bleu", score=score, passed=passed),
        }
        for index, (score, passed) in scores.items()
    }
    summary = {
        "bleu": sum(score for score, _ in scores.values()) / len(scores) if scores else 0.0
    }
    return Run(id=run_id, dataset="d", name=run_id, results=results, summary=summary)


def test_comparison_identical_runs() -> None:
    baseline = _run("a", {0: (0.5, True), 1: (0.8, True)})
    result = compare_runs(baseline, baseline)
    assert result.counts() == {"regressions": 0, "improvements": 0, "fixed": 0, "broken": 0}


def test_comparison_detects_improvement() -> None:
    baseline = _run("a", {0: (0.4, False)})
    candidate = _run("b", {0: (0.9, True)})
    result = compare_runs(baseline, candidate)
    assert len(result.improvements) == 1
    assert len(result.fixed) == 1


def test_comparison_detects_regression() -> None:
    baseline = _run("a", {0: (0.9, True)})
    candidate = _run("b", {0: (0.2, False)})
    result = compare_runs(baseline, candidate)
    assert len(result.regressions) == 1
    assert len(result.broken) == 1
    assert result.has_regressions() is True


def test_comparison_scores_delta() -> None:
    baseline = _run("a", {0: (0.5, True)})
    candidate = _run("b", {0: (0.7, True)})
    result = compare_runs(baseline, candidate)
    assert result.scores["bleu"]["delta"] == round(0.7 - 0.5, 10)


def test_comparison_handles_missing_examples() -> None:
    baseline = _run("a", {0: (0.5, True), 1: (0.5, True)})
    candidate = _run("b", {0: (0.5, True)})
    result = compare_runs(baseline, candidate)
    assert any(delta.example_index == 1 for delta in result.deltas)


def test_comparison_ids_recorded() -> None:
    baseline = _run("a", {0: (0.5, True)})
    candidate = _run("b", {0: (0.5, True)})
    result = compare_runs(baseline, candidate)
    assert result.baseline_id == "a"
    assert result.candidate_id == "b"
