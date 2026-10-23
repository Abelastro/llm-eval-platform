"""Faithfulness metric: claim-level groundedness in the retrieved context."""

from __future__ import annotations

import re

from llm_eval.metrics.base import Metric, token_set
from llm_eval.models import Example, MetricResult

_SENTENCE_RE = re.compile(r"[.!?]+")


def split_claims(text: str) -> list[str]:
    """Split an answer into non-empty claim sentences.

    Args:
        text: Answer text.

    Returns:
        A list of trimmed, non-empty claim strings.
    """

    if not text or not text.strip():
        return []
    parts = _SENTENCE_RE.split(text)
    claims = [part.strip() for part in parts if part.strip()]
    return claims or [text.strip()]


class FaithfulnessMetric(Metric):
    """Measure the fraction of answer claims grounded in the context.

    Each claim is considered supported when at least ``support_threshold`` of
    its unique tokens appear in the union of context tokens. The score is the
    supported-claim fraction.
    """

    name = "faithfulness"
    threshold = 0.5

    def __init__(self, support_threshold: float = 0.6,
                 threshold: float | None = None) -> None:
        """Initialize the metric.

        Args:
            support_threshold: Token-coverage ratio required to support a claim.
            threshold: Override the pass threshold.
        """

        super().__init__(threshold)
        self.support_threshold = support_threshold

    def _claim_coverage(self, claim: str, context_tokens: set[str]) -> float:
        """Compute the fraction of claim tokens present in the context.

        Args:
            claim: Claim sentence.
            context_tokens: Union of context tokens.

        Returns:
            Coverage ratio in ``[0, 1]``.
        """

        claim_tokens = token_set(claim)
        if not claim_tokens:
            return 0.0
        return len(claim_tokens & context_tokens) / len(claim_tokens)

    def measure(self, example: Example) -> MetricResult:
        """Compute faithfulness for the example.

        Args:
            example: Example with ``actual`` as the answer and ``context`` as
                the retrieved passages.

        Returns:
            Supported-claim fraction in ``[0, 1]``.
        """

        if not example.context:
            return self._result(0.0, reason="no context provided to ground against")

        claims = split_claims(example.actual)
        if not claims:
            return self._result(0.0, reason="answer contains no claims")

        context_tokens = token_set(" ".join(example.context))
        coverages = [self._claim_coverage(claim, context_tokens) for claim in claims]
        supported = [c >= self.support_threshold for c in coverages]
        score = sum(supported) / len(claims)
        return self._result(
            score,
            reason=f"{sum(supported)}/{len(claims)} claims grounded in context",
            metadata={
                "claims": claims,
                "coverages": coverages,
                "support_threshold": self.support_threshold,
            },
        )
