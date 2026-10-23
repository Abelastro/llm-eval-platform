"""ROUGE-L metric based on longest common subsequence."""

from __future__ import annotations

from llm_eval.metrics.base import Metric, lcs_length, tokenize
from llm_eval.models import Example, MetricResult


class RougeLMetric(Metric):
    """ROUGE-L F-measure computed from the longest common subsequence.

    The F-measure uses ``F = (1 + beta^2) * P * R / (R + beta^2 * P)`` where the
    default ``beta`` of ``1.0`` weights precision and recall equally. Recall is
    defined over the reference tokens and precision over the candidate tokens.
    """

    name = "rouge_l"
    threshold = 0.4

    def __init__(self, beta: float = 1.0, threshold: float | None = None) -> None:
        """Initialize the metric.

        Args:
            beta: Recall/precision weighting factor.
            threshold: Override the pass threshold.
        """

        super().__init__(threshold)
        self.beta = max(0.0, beta)

    def measure(self, example: Example) -> MetricResult:
        """Compute ROUGE-L for the example.

        Args:
            example: Example with ``actual`` as candidate and ``expected`` as
                reference.

        Returns:
            ROUGE-L F-measure in ``[0, 1]``.
        """

        if example.expected is None:
            return self._result(0.0, reason="no expected answer provided")

        candidate = tokenize(example.actual)
        reference = tokenize(example.expected)
        if not candidate or not reference:
            return self._result(0.0, reason="candidate or reference is empty")

        lcs = lcs_length(candidate, reference)
        recall = lcs / len(reference)
        precision = lcs / len(candidate)
        beta_sq = self.beta ** 2
        if precision + recall == 0.0:
            f_measure = 0.0
        else:
            f_measure = (1 + beta_sq) * precision * recall / (recall + beta_sq * precision)

        return self._result(
            f_measure,
            reason=f"LCS length {lcs} (precision {precision:.3f}, recall {recall:.3f})",
            metadata={"lcs": lcs, "precision": precision, "recall": recall},
        )
