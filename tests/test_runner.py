"""Tests for the evaluation runner."""

from __future__ import annotations

import pytest

from llm_eval.metrics.exact_match import ExactMatchMetric
from llm_eval.metrics.registry import MetricRegistry
from llm_eval.models import Dataset, Example, Run
from llm_eval.runners.runner import Runner, run_dataset, summarize


def test_runner_requires_metrics() -> None:
    with pytest.raises(ValueError):
        Runner([])


def test_runner_produces_run(sample_dataset: Dataset) -> None:
    run = Runner([ExactMatchMetric()]).run(sample_dataset, name="r1")
    assert isinstance(run, Run)
    assert run.dataset == "sample"
    assert run.name == "r1"
    assert set(run.results) == {0, 1, 2}


def test_runner_records_per_metric_results(sample_dataset: Dataset) -> None:
    run = run_dataset(sample_dataset, ["exact_match", "contains"])
    assert "exact_match" in run.results[0]
    assert "contains" in run.results[0]


def test_runner_summary_contains_overall(sample_dataset: Dataset) -> None:
    run = run_dataset(sample_dataset, ["exact_match"])
    assert "overall_pass_rate" in run.summary
    assert 0.0 <= run.summary["overall_pass_rate"] <= 1.0


def test_from_names_requires_names() -> None:
    with pytest.raises(ValueError):
        Runner.from_names([])


def test_from_names_unknown_metric() -> None:
    with pytest.raises(KeyError):
        Runner.from_names(["nope"])


def test_run_pass_rate_overall(sample_dataset: Dataset) -> None:
    run = run_dataset(sample_dataset, ["exact_match", "contains"])
    assert 0.0 <= run.pass_rate() <= 1.0


def test_run_pass_rate_specific_metric(sample_dataset: Dataset) -> None:
    run = run_dataset(sample_dataset, ["exact_match"])
    assert run.pass_rate("exact_match") == run.pass_rate()


def test_run_pass_rate_empty() -> None:
    run = Run(dataset="empty")
    assert run.pass_rate() == 0.0


def test_summarize_empty_results() -> None:
    summary = summarize({}, [ExactMatchMetric()])
    assert summary["exact_match"] == 0.0
    assert summary["overall_pass_rate"] == 0.0


def test_runner_with_custom_registry(sample_dataset: Dataset) -> None:
    registry = MetricRegistry()
    run = Runner.from_names(["bleu"], registry=registry).run(sample_dataset)
    assert "bleu" in run.summary
