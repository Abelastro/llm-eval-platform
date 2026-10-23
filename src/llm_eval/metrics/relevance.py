"""Relevance metric: symmetric answer-context token overlap."""

from __future__ import annotations

from llm_eval.metrics.base import Metric, token_set
from llm_eval.models import Example, MetricResult


class RelevanceMetric(Metric):
    """Measure answer relevance as the F1 of answer and context token sets.

    Precision is the fraction of answer tokens present in the context and
    recall is the fraction of context tokens present in the answer. The score
    is their harmonic mean, which rewards concise answers that reuse the
    retrieved vocabulary.
    """

    name = "relevance"
    threshold = 0.4

    def __init__(self, include_input: bool = False,
                 threshold: float | None = None) -> None:
        """Initialize the metric.

        Args:
            include_input: Fold the question tokens into the answer vocabulary.
            threshold: Override the pass threshold.
        """

        super().__init__(threshold)
        self.include_input = include_input

    def measure(self, example: Example) -> MetricResult:
        """Compute answer/context relevance for the example.

        Args:
            example: Example with ``actual`` as the answer and ``context`` as
                the retrieved passages.

        Returns:
            F1 overlap in ``[0, 1]``.
        """

        if not example.context:
            return self._result(0.0, reason="no context provided to compare against")

        answer_text = example.actual
        if self.include_input and example.input:
            answer_text = f"{example.input} {example.actual}"
        answer = token_set(answer_text)
        context = token_set(" ".join(example.context))
        if not answer or not context:
            return self._result(0.0, reason="answer or context produced no tokens")

        overlap = len(answer & context)
        precision = overlap / len(answer)
        recall = overlap / len(context)
        if precision + recall == 0.0:
            f_measure = 0.0
        else:
            f_measure = 2 * precision * recall / (precision + recall)

        return self._result(
            f_measure,
            reason=f"overlap {overlap} tokens (precision {precision:.3f}, "
            f"recall {recall:.3f})",
            metadata={"overlap": overlap, "precision": precision, "recall": recall},
        )
