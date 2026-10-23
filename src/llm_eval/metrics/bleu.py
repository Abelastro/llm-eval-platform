"""BLEU-4 metric implemented from scratch.

The implementation follows the standard sentence-level BLEU formulation:

1. For each n in 1..N compute the clipped modified precision of the candidate
   n-grams against the reference n-grams.
2. Apply add-one smoothing to orders whose clipped count is zero.
3. Take the geometric mean of the available precisions.
4. Multiply by the brevity penalty ``exp(1 - r/c)`` when the candidate is
   shorter than the reference.

With ``effective_order`` enabled (the default) only orders up to the candidate
length are considered, which keeps short answers from being penalized twice.
"""

from __future__ import annotations

import math

from llm_eval.metrics.base import Metric, ngrams, tokenize
from llm_eval.models import Example, MetricResult

_MAX_ORDER = 4


class BLEUMetric(Metric):
    """Sentence BLEU score with add-one smoothing."""

    name = "bleu"
    threshold = 0.3

    def __init__(self, max_order: int = _MAX_ORDER, effective_order: bool = True,
                 threshold: float | None = None) -> None:
        """Initialize the metric.

        Args:
            max_order: Highest n-gram order (default BLEU-4).
            effective_order: Restrict orders to the candidate length.
            threshold: Override the pass threshold.
        """

        super().__init__(threshold)
        self.max_order = max(1, max_order)
        self.effective_order = effective_order

    @staticmethod
    def _modified_precision(candidate: list[str], reference: list[str],
                            order: int) -> tuple[int, int]:
        """Compute clipped matches and total candidate n-grams for one order.

        Args:
            candidate: Candidate token sequence.
            reference: Reference token sequence.
            order: n-gram order.

        Returns:
            A ``(clipped, total)`` tuple.
        """

        candidate_ngrams = ngrams(candidate, order)
        if not candidate_ngrams:
            return 0, 0
        reference_ngrams = ngrams(reference, order)
        clipped = sum(
            min(count, reference_ngrams.get(gram, 0))
            for gram, count in candidate_ngrams.items()
        )
        return clipped, sum(candidate_ngrams.values())

    def measure(self, example: Example) -> MetricResult:
        """Compute sentence BLEU-4 for the example.

        Args:
            example: Example with ``actual`` as candidate and ``expected`` as
                reference.

        Returns:
            BLEU score in ``[0, 1]``.
        """

        if example.expected is None:
            return self._result(0.0, reason="no expected answer provided")

        candidate = tokenize(example.actual)
        reference = tokenize(example.expected)
        if not candidate or not reference:
            return self._result(0.0, reason="candidate or reference is empty")

        max_order = self.max_order
        if self.effective_order:
            max_order = min(max_order, len(candidate))

        log_sum = 0.0
        orders_used = 0
        for order in range(1, max_order + 1):
            clipped, total = self._modified_precision(candidate, reference, order)
            if total == 0:
                continue
            if clipped == 0:
                clipped += 1
                total += 1
            log_sum += math.log(clipped / total)
            orders_used += 1

        if orders_used == 0:
            return self._result(0.0, reason="no n-gram orders available")

        geo_mean = math.exp(log_sum / orders_used)
        candidate_len = len(candidate)
        reference_len = len(reference)
        if candidate_len > reference_len:
            brevity = 1.0
        else:
            brevity = math.exp(1.0 - reference_len / candidate_len)

        score = geo_mean * brevity
        return self._result(
            score,
            reason=f"BLEU-{orders_used} geometric mean {geo_mean:.4f}, "
            f"brevity penalty {brevity:.4f}",
            metadata={"orders": orders_used, "brevity_penalty": brevity},
        )
