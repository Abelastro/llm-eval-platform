"""Tests for the metric registry and plugin discovery."""

from __future__ import annotations

import pytest

from llm_eval.metrics.base import Metric
from llm_eval.metrics.registry import MetricRegistry, get_registry
from llm_eval.models import Example, MetricResult


class DummyMetric(Metric):
    """A trivial metric used to exercise registration paths."""

    name = "dummy"

    def measure(self, example: Example) -> MetricResult:
        return self._result(1.0, reason="dummy")


def test_builtins_are_registered() -> None:
    registry = MetricRegistry()
    names = registry.list()
    for expected in {"exact_match", "contains", "bleu", "rouge_l", "faithfulness", "relevance"}:
        assert expected in names


def test_create_returns_metric_instance() -> None:
    metric = MetricRegistry().create("bleu")
    assert isinstance(metric, Metric)
    assert metric.name == "bleu"


def test_create_unknown_raises() -> None:
    with pytest.raises(KeyError):
        MetricRegistry().create("does_not_exist")


def test_register_custom_factory() -> None:
    registry = MetricRegistry()
    registry.register("dummy", DummyMetric)
    assert "dummy" in registry
    assert isinstance(registry.create("dummy"), DummyMetric)


def test_register_alias() -> None:
    registry = MetricRegistry()
    registry.register("dummy", DummyMetric, aliases=["d"])
    assert "d" in registry
    assert registry.create("d").name == "dummy"


def test_create_with_attribute_override() -> None:
    metric = MetricRegistry().create("bleu", threshold=0.9)
    assert metric.threshold == 0.9


def test_create_with_unknown_attribute_raises() -> None:
    with pytest.raises(KeyError):
        MetricRegistry().create("bleu", nope=1)


def test_contains_rejects_non_string() -> None:
    assert 123 not in MetricRegistry()


def test_get_registry_singleton() -> None:
    assert get_registry() is get_registry()


def test_discover_plugins_returns_list() -> None:
    registry = MetricRegistry()
    discovered = registry.discover_plugins(group="llm_eval.nonexistent")
    assert discovered == []
