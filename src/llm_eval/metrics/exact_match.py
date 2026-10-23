"""Exact-match metric."""

from __future__ import annotations

import re

from llm_eval.metrics.base import Metric
from llm_eval.models import Example, MetricResult

_WHITESPACE_RE = re.compile(r"\s+")


class ExactMatchMetric(Metric):
    """Binary metric that scores normalized string equality.

    Normalization lowercases the text and collapses runs of whitespace. When
    ``case_sensitive`` is enabled only whitespace is normalized.
    """

    name = "exact_match"
    threshold = 1.0

    def __init__(self, case_sensitive: bool = False, threshold: float | None = None) -> None:
        """Initialize the metric.

        Args:
            case_sensitive: Preserve case when comparing strings.
            threshold: Override the pass threshold.
        """

        super().__init__(threshold)
        self.case_sensitive = case_sensitive

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison.

        Args:
            text: Text to normalize.

        Returns:
            Whitespace-collapsed, optionally lowercased text.
        """

        collapsed = _WHITESPACE_RE.sub(" ", text or "").strip()
        return collapsed if self.case_sensitive else collapsed.lower()

    def measure(self, example: Example) -> MetricResult:
        """Compare ``actual`` to ``expected`` for normalized equality.

        Args:
            example: Example to evaluate.

        Returns:
            Score ``1.0`` on a match and ``0.0`` otherwise.
        """

        if example.expected is None:
            return self._result(0.0, reason="no expected answer provided")
        match = self._normalize(example.actual) == self._normalize(example.expected)
        return self._result(
            1.0 if match else 0.0,
            reason="normalized strings match" if match else "normalized strings differ",
            metadata={"case_sensitive": self.case_sensitive},
        )
