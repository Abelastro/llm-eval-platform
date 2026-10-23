"""Tests for the CI quality gate."""

from __future__ import annotations

from llm_eval.gates.quality_gate import QualityGate, evaluate_gate
from llm_eval.models import MetricResult, Run
from llm_eval.runners.compare import compare_runs


def _run(run_id: str = "r", bleu: float = 0.5, passed: bool = True) -> Run:
    results = {
        0: {"bleu": MetricResult(name="bleu", score=bleu, passed=passed)},
        1: {"bleu": MetricResult(name="bleu", score=bleu, passed=passed)},
    }
    return Run(
        id=run_id, dataset="d", name=run_id, results=results, summary={"bleu": bleu}
    )


def test_gate_passes_when_thresholds_met() -> None:
    result = QualityGate(metric_thresholds={"bleu": 0.3}, overall_pass_rate=0.5).evaluate(
        _run(bleu=0.5, passed=True)
    )
    assert result.passed is True


def test_gate_fails_when_below_threshold() -> None:
    result = QualityGate(metric_thresholds={"bleu": 0.9}).evaluate(_run(bleu=0.5))
    assert result.passed is False
    assert any("below threshold" in reason for reason in result.reasons)


def test_gate_fails_on_missing_metric() -> None:
    result = QualityGate(metric_thresholds={"rouge_l": 0.1}).evaluate(_run())
    assert result.passed is False
    assert any("not present" in reason for reason in result.reasons)


def test_gate_fails_on_overall_pass_rate() -> None:
    result = QualityGate(overall_pass_rate=0.9).evaluate(_run(bleu=0.1, passed=False))
    assert result.passed is False


def test_gate_records_observed_metrics() -> None:
    result = QualityGate(metric_thresholds={"bleu": 0.3}).evaluate(_run(bleu=0.7))
    assert result.metrics["bleu"] == 0.7


def test_evaluate_gate_convenience() -> None:
    result = evaluate_gate(_run(bleu=0.8), {"bleu": 0.5}, overall_pass_rate=0.5)
    assert result.passed is True


def test_gate_comparison_allows_budget() -> None:
    baseline = _run("a", bleu=0.9, passed=True)
    candidate = _run("b", bleu=0.1, passed=False)
    comparison = compare_runs(baseline, candidate)
    gate = QualityGate(max_regressions=1)
    result = gate.evaluate_comparison(comparison)
    assert result.passed is False
    assert result.metrics["broken"] == 2.0


def test_gate_comparison_pass_within_budget() -> None:
    baseline = _run("a", bleu=0.5, passed=True)
    candidate = _run("b", bleu=0.5, passed=True)
    comparison = compare_runs(baseline, candidate)
    result = QualityGate(max_regressions=0).evaluate_comparison(comparison)
    assert result.passed is True
