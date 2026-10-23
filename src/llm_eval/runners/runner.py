"""Execution engine that scores datasets with metrics."""

from __future__ import annotations

from typing import Sequence

from llm_eval.metrics.base import Metric
from llm_eval.metrics.registry import MetricRegistry, get_registry
from llm_eval.models import Dataset, MetricResult, Run


class Runner:
    """Evaluate a dataset with an ordered set of metrics."""

    def __init__(self, metrics: Sequence[Metric]) -> None:
        """Initialize the runner.

        Args:
            metrics: Metrics to apply to every example.

        Raises:
            ValueError: If ``metrics`` is empty.
        """

        if not metrics:
            raise ValueError("at least one metric is required")
        self.metrics: list[Metric] = list(metrics)

    @classmethod
    def from_names(cls, names: Sequence[str],
                   registry: MetricRegistry | None = None,
                   **metric_kwargs: object) -> "Runner":
        """Build a runner from registered metric names.

        Args:
            names: Metric names to instantiate.
            registry: Registry to resolve names; defaults to the global one.
            **metric_kwargs: Forwarded to each metric constructor.

        Returns:
            A configured :class:`Runner`.

        Raises:
            KeyError: If a name is not registered.
            ValueError: If ``names`` is empty.
        """

        if not names:
            raise ValueError("at least one metric name is required")
        active = registry or get_registry()
        metrics = [active.create(name, **metric_kwargs) for name in names]
        return cls(metrics)

    def run(self, dataset: Dataset, name: str = "run") -> Run:
        """Score every example in ``dataset``.

        Args:
            dataset: Dataset to evaluate.
            name: Human-readable run label.

        Returns:
            A :class:`Run` containing per-example results and summaries.
        """

        results: dict[int, dict[str, MetricResult]] = {}
        for index, example in enumerate(dataset.examples):
            per_example: dict[str, MetricResult] = {}
            for metric in self.metrics:
                per_example[metric.name] = metric.measure(example)
            results[index] = per_example

        run = Run(
            dataset=dataset.name,
            name=name,
            results=results,
            summary=summarize(results, self.metrics),
        )
        return run


def summarize(results: dict[int, dict[str, MetricResult]],
              metrics: Sequence[Metric]) -> dict[str, float]:
    """Aggregate per-example results into per-metric summary statistics.

    Args:
        results: Per-example metric results.
        metrics: Metrics that were applied.

    Returns:
        Mapping of metric name to mean score, plus
        ``overall_pass_rate`` across every metric result.
    """

    summary: dict[str, float] = {}
    for metric in metrics:
        scores = [
            per_example[metric.name].score
            for per_example in results.values()
            if metric.name in per_example
        ]
        summary[metric.name] = sum(scores) / len(scores) if scores else 0.0

    total = 0
    passed = 0
    for per_example in results.values():
        for result in per_example.values():
            total += 1
            if result.passed:
                passed += 1
    summary["overall_pass_rate"] = passed / total if total else 0.0
    return summary


def run_dataset(dataset: Dataset, names: Sequence[str], name: str = "run",
                registry: MetricRegistry | None = None) -> Run:
    """Convenience helper that builds a runner and evaluates a dataset.

    Args:
        dataset: Dataset to evaluate.
        names: Metric names to apply.
        name: Human-readable run label.
        registry: Registry used to resolve metric names.

    Returns:
        The resulting :class:`Run`.
    """

    return Runner.from_names(names, registry=registry).run(dataset, name=name)
