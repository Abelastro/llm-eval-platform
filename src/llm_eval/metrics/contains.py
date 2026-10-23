"""Contains metric: substring and token containment."""

from __future__ import annotations

import re

from llm_eval.metrics.base import Metric, token_set
from llm_eval.models import Example, MetricResult

_WHITESPACE_RE = re.compile(r"\s+")


class ContainsMetric(Metric):
    """Score whether ``actual`` contains the reference ``expected`` answer.

    In ``substring`` mode the metric performs a normalized substring check. In
    ``tokens`` mode it reports the fraction of reference tokens present in the
    actual answer, which gives partial credit for paraphrases.
    """

    name = "contains"
    threshold = 0.5

    def __init__(self, mode: str = "substring", case_sensitive: bool = False,
                 threshold: float | None = None) -> None:
        """Initialize the metric.

        Args:
            mode: Either ``"substring"`` or ``"tokens"``.
            case_sensitive: Preserve case during normalization.
            threshold: Override the pass threshold.

        Raises:
            ValueError: If ``mode`` is not recognized.
        """

        super().__init__(threshold)
        if mode not in {"substring", "tokens"}:
            raise ValueError("mode must be 'substring' or 'tokens'")
        self.mode = mode
        self.case_sensitive = case_sensitive

    def _normalize(self, text: str) -> str:
        """Normalize whitespace and optionally case.

        Args:
            text: Text to normalize.

        Returns:
            The normalized text.
        """

        collapsed = _WHITESPACE_RE.sub(" ", text or "").strip()
        return collapsed if self.case_sensitive else collapsed.lower()

    def measure(self, example: Example) -> MetricResult:
        """Score containment of ``expected`` within ``actual``.

        Args:
            example: Example to evaluate.

        Returns:
            A score in ``[0, 1]`` with diagnostics.
        """

        if example.expected is None or not example.expected.strip():
            return self._result(0.0, reason="no expected answer provided")

        if self.mode == "substring":
            haystack = self._normalize(example.actual)
            needle = self._normalize(example.expected)
            contained = needle in haystack
            return self._result(
                1.0 if contained else 0.0,
                reason="expected is a substring of actual" if contained
                else "expected is not a substring of actual",
                metadata={"mode": self.mode},
            )

        reference = token_set(example.expected if self.case_sensitive
                              else example.expected.lower())
        actual = token_set(example.actual if self.case_sensitive
                           else example.actual.lower())
        if not reference:
            return self._result(0.0, reason="expected produced no tokens")
        overlap = len(reference & actual) / len(reference)
        return self._result(
            overlap,
            reason=f"{overlap:.2%} of reference tokens present in actual",
            metadata={"mode": self.mode, "missing": sorted(reference - actual)},
        )
