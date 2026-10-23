"""Metric base types and shared text utilities."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Iterable

from llm_eval.models import Example, MetricResult

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric word tokens.

    Args:
        text: Input text, possibly ``None``-like.

    Returns:
        A list of lowercase tokens; empty for falsy input.
    """

    if not text:
        return []
    return _WORD_RE.findall(text.lower())


def token_set(text: str) -> set[str]:
    """Return the unique token set of ``text``.

    Args:
        text: Input text.

    Returns:
        A set of unique lowercase tokens.
    """

    return set(tokenize(text))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``.

    Args:
        value: Value to clamp.
        low: Lower bound.
        high: Upper bound.

    Returns:
        The clamped value.
    """

    return max(low, min(high, value))


def lcs_length(left: list[str], right: list[str]) -> int:
    """Compute the longest common subsequence length of two token lists.

    Args:
        left: First token sequence.
        right: Second token sequence.

    Returns:
        Length of the longest common subsequence.
    """

    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for token_left in left:
        current = [0]
        for index, token_right in enumerate(right, start=1):
            if token_left == token_right:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    """Count n-grams in a token list.

    Args:
        tokens: Token sequence.
        n: Gram size.

    Returns:
        Mapping of n-gram tuples to occurrence counts; empty when the sequence
        is shorter than ``n``.
    """

    counts: dict[tuple[str, ...], int] = {}
    if n <= 0 or len(tokens) < n:
        return counts
    for index in range(len(tokens) - n + 1):
        gram = tuple(tokens[index : index + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


class Metric(ABC):
    """Abstract base class for all evaluation metrics.

    Subclasses declare a unique ``name`` and implement :meth:`measure`. The
    optional ``threshold`` drives the ``passed`` flag on each result.
    """

    name: str = "metric"
    threshold: float = 0.5

    def __init__(self, threshold: float | None = None) -> None:
        """Initialize the metric.

        Args:
            threshold: Override the class-level pass threshold.
        """

        if threshold is not None:
            self.threshold = threshold

    @abstractmethod
    def measure(self, example: Example) -> MetricResult:
        """Score a single example.

        Args:
            example: The example to evaluate.

        Returns:
            A populated :class:`MetricResult`.
        """

        raise NotImplementedError

    def passes(self, score: float) -> bool:
        """Return whether a score meets the metric threshold.

        Args:
            score: Candidate score.

        Returns:
            ``True`` when ``score >= threshold``.
        """

        return score >= self.threshold

    def _result(
        self,
        score: float,
        reason: str = "",
        metadata: dict | None = None,
    ) -> MetricResult:
        """Construct a :class:`MetricResult` with the metric's threshold.

        Args:
            score: Computed score.
            reason: Explanation of the score.
            metadata: Optional diagnostic payload.

        Returns:
            The assembled metric result.
        """

        return MetricResult(
            name=self.name,
            score=float(score),
            passed=self.passes(float(score)),
            reason=reason,
            metadata=dict(metadata or {}),
        )

    def measure_many(self, examples: Iterable[Example]) -> list[MetricResult]:
        """Score an iterable of examples.

        Args:
            examples: Examples to evaluate.

        Returns:
            A list of results in input order.
        """

        return [self.measure(example) for example in examples]

    def __repr__(self) -> str:
        """Return an unambiguous representation of the metric."""

        return f"{type(self).__name__}(name={self.name!r}, threshold={self.threshold!r})"
