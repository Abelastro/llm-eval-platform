"""Metric registry with entry-point plugin discovery.

Built-in metrics are registered lazily on first use. Third-party metrics can be
exposed to the platform by advertising a ``llm_eval.metrics`` entry point whose
value resolves to a :class:`~llm_eval.metrics.base.Metric` subclass or factory.
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from typing import Callable, Iterable

from llm_eval.metrics.base import Metric

MetricFactory = Callable[[], Metric]

_BUILTINS = {
    "exact_match": ("llm_eval.metrics.exact_match", "ExactMatchMetric"),
    "contains": ("llm_eval.metrics.contains", "ContainsMetric"),
    "bleu": ("llm_eval.metrics.bleu", "BLEUMetric"),
    "rouge_l": ("llm_eval.metrics.rouge", "RougeLMetric"),
    "faithfulness": ("llm_eval.metrics.faithfulness", "FaithfulnessMetric"),
    "relevance": ("llm_eval.metrics.relevance", "RelevanceMetric"),
}


class MetricRegistry:
    """Registry mapping metric names to factories."""

    def __init__(self) -> None:
        """Initialize an empty registry."""

        self._factories: dict[str, MetricFactory] = {}
        self._builtins_loaded = False

    def register(self, name: str, factory: MetricFactory,
                 *, aliases: Iterable[str] = ()) -> None:
        """Register a metric factory.

        Args:
            name: Canonical metric name.
            factory: Zero-argument callable returning a metric instance.
            aliases: Additional names that resolve to the same factory.
        """

        self._factories[name] = factory
        for alias in aliases:
            self._factories[alias] = factory

    def _load_builtins(self) -> None:
        """Import and register the built-in metrics on first use."""

        if self._builtins_loaded:
            return
        self._builtins_loaded = True
        for name, (module, attribute) in _BUILTINS.items():
            if name in self._factories:
                continue
            self._factories[name] = _lazy_factory(module, attribute)

    def discover_plugins(self, group: str = "llm_eval.metrics") -> list[str]:
        """Discover and register metrics advertised via entry points.

        Args:
            group: Entry-point group to scan.

        Returns:
            Sorted names of plugin metrics that were registered.
        """

        self._load_builtins()
        discovered: list[str] = []
        try:
            entry_points = importlib_metadata.entry_points()
            if hasattr(entry_points, "select"):
                selected = entry_points.select(group=group)
            else:
                selected = entry_points.get(group, [])  # type: ignore[attr-defined]
        except Exception:
            return discovered

        for entry_point in selected:
            try:
                loaded = entry_point.load()
            except Exception:
                continue
            factory = _coerce_factory(loaded)
            if factory is None:
                continue
            self.register(entry_point.name, factory)
            discovered.append(entry_point.name)
        return sorted(discovered)

    def create(self, name: str, **kwargs: object) -> Metric:
        """Instantiate a registered metric.

        Args:
            name: Registered metric name.
            **kwargs: Forwarded to the metric constructor.

        Returns:
            A metric instance.

        Raises:
            KeyError: If the name is not registered.
        """

        self._load_builtins()
        factory = self._factories.get(name)
        if factory is None:
            raise KeyError(f"unknown metric: {name}")
        metric = factory()
        for key, value in kwargs.items():
            if not hasattr(metric, key):
                raise KeyError(f"metric {name} has no attribute {key!r}")
            setattr(metric, key, value)
        return metric

    def get(self, name: str):
        """Return the registered factory for a metric.

        Args:
            name: Registered metric name.

        Returns:
            The metric factory.

        Raises:
            KeyError: If the name is not registered.
        """

        self._load_builtins()
        if name not in self._factories:
            raise KeyError(f"unknown metric: {name}")
        return self._factories[name]

    def list(self) -> list[str]:
        """List registered metric names.

        Returns:
            Sorted metric names.
        """

        self._load_builtins()
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        """Return whether a metric name is registered."""

        if not isinstance(name, str):
            return False
        self._load_builtins()
        return name in self._factories


def _lazy_factory(module: str, attribute: str) -> MetricFactory:
    """Build a factory that imports its target lazily.

    Args:
        module: Dotted module path.
        attribute: Attribute name within the module.

    Returns:
        A callable returning the imported metric instance.
    """

    def factory() -> Metric:
        import importlib

        imported = importlib.import_module(module)
        target = getattr(imported, attribute)
        return target()

    return factory


def _coerce_factory(loaded: object) -> MetricFactory | None:
    """Normalize an entry-point payload into a metric factory.

    Args:
        loaded: The object returned by ``entry_point.load()``.

    Returns:
        A callable producing a metric, or ``None`` when the payload is not a
        metric class or factory.
    """

    if isinstance(loaded, type) and issubclass(loaded, Metric):
        return loaded  # type: ignore[return-value]
    if callable(loaded):
        return loaded  # type: ignore[return-value]
    return None


_REGISTRY = MetricRegistry()


def get_registry() -> MetricRegistry:
    """Return the process-wide metric registry.

    Returns:
        The shared :class:`MetricRegistry` singleton.
    """

    return _REGISTRY
