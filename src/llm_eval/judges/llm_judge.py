"""LLM-as-judge scoring with pluggable clients.

The platform ships with a deterministic :class:`HeuristicJudgeClient` that needs
no network access. A :class:`OpenAIChatJudgeClient` stub is provided for real
deployments but is only instantiated when an API key is configured, keeping the
default execution path fully offline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from llm_eval.judges.rubric import Rubric, RubricCriterion, default_rubric
from llm_eval.metrics.base import Metric, clamp, token_set
from llm_eval.models import Example, MetricResult


@runtime_checkable
class JudgeClient(Protocol):
    """Protocol implemented by judge backends."""

    def score(self, criterion: RubricCriterion, example: Example) -> float:
        """Score one criterion for one example.

        Args:
            criterion: The rubric criterion being judged.
            example: The example under evaluation.

        Returns:
            A score in ``[0, 1]``.
        """

        ...


class HeuristicJudgeClient:
    """Deterministic offline judge based on lexical overlap and heuristics.

    The client is intentionally simple so that evaluation runs are reproducible
    and require no API keys. It approximates a rubric-aware judge using token
    overlap for correctness/completeness and structural cues for clarity.
    """

    def __init__(self, support_threshold: float = 0.6) -> None:
        """Initialize the client.

        Args:
            support_threshold: Coverage ratio treated as full support.
        """

        self.support_threshold = max(1e-6, support_threshold)

    def _correctness(self, example: Example) -> float:
        """Estimate correctness from answer/reference token overlap.

        Args:
            example: Example under evaluation.

        Returns:
            Overlap score in ``[0, 1]``.
        """

        if not example.expected:
            return 0.0
        expected = token_set(example.expected)
        actual = token_set(example.actual)
        if not expected:
            return 0.0
        return len(expected & actual) / len(expected)

    def _completeness(self, example: Example) -> float:
        """Estimate completeness from reference-token coverage.

        Args:
            example: Example under evaluation.

        Returns:
            Coverage score in ``[0, 1]``.
        """

        return self._correctness(example)

    def _clarity(self, example: Example) -> float:
        """Estimate clarity from answer length and sentence structure.

        Args:
            example: Example under evaluation.

        Returns:
            Clarity score in ``[0, 1]``.
        """

        tokens = token_set(example.actual)
        if not tokens:
            return 0.0
        words = len(example.actual.split())
        length_score = clamp(words / 12.0)
        sentence_score = 1.0 if example.actual.strip().endswith((".", "!", "?")) else 0.7
        return clamp(0.6 * length_score + 0.4 * sentence_score)

    def score(self, criterion: RubricCriterion, example: Example) -> float:
        """Score a criterion using the heuristic matching its name.

        Args:
            criterion: Rubric criterion being judged.
            example: Example under evaluation.

        Returns:
            A score in ``[0, 1]``.
        """

        name = criterion.name.lower()
        if "correct" in name or "accuracy" in name or "factual" in name:
            return clamp(self._correctness(example))
        if "complete" in name or "coverage" in name or "recall" in name:
            return clamp(self._completeness(example))
        if "clarity" in name or "readab" in name or "structure" in name:
            return clamp(self._clarity(example))
        return clamp(self._correctness(example))


class OpenAIChatJudgeClient:
    """Stub judge client for an OpenAI-compatible chat endpoint.

    The client never performs network I/O in this package. It exists so that
    production deployments can supply credentials and an HTTP transport without
    touching the evaluation core. Instantiate it only when an API key is
    available.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1") -> None:
        """Initialize the client.

        Args:
            api_key: [REDACTED] API key.
            model: Chat model identifier.
            base_url: API base URL.

        Raises:
            ValueError: If ``api_key`` is empty.
        """

        if not api_key:
            raise ValueError("api_key must be provided for OpenAIChatJudgeClient")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def score(self, criterion: RubricCriterion, example: Example) -> float:
        """Score a criterion against a remote chat model.

        Args:
            criterion: Rubric criterion being judged.
            example: Example under evaluation.

        Returns:
            A score in ``[0, 1]``.

        Raises:
            RuntimeError: Always, because this stub performs no network I/O.
        """

        raise RuntimeError(
            "OpenAIChatJudgeClient is a network stub; provide a real transport "
            "implementation before use"
        )


class LLMJudge(Metric):
    """Weighted rubric judge that delegates per-criterion scoring to a client."""

    name = "llm_judge"
    threshold = 0.5

    def __init__(self, client: JudgeClient | None = None,
                 rubric: Rubric | None = None, threshold: float | None = None) -> None:
        """Initialize the judge.

        Args:
            client: Judge backend; defaults to :class:`HeuristicJudgeClient`.
            rubric: Rubric to apply; defaults to :func:`default_rubric`.
            threshold: Override the pass threshold.
        """

        super().__init__(threshold)
        self.client: JudgeClient = client or HeuristicJudgeClient()
        self.rubric = rubric or default_rubric()

    def measure(self, example: Example) -> MetricResult:
        """Score the example against every rubric criterion.

        Args:
            example: Example under evaluation.

        Returns:
            The weighted aggregate score in ``[0, 1]``.
        """

        weights = self.rubric.normalized_weights()
        if not weights:
            return self._result(0.0, reason="rubric has no weighted criteria")

        per_criterion: dict[str, float] = {}
        for criterion in self.rubric.criteria:
            per_criterion[criterion.name] = clamp(self.client.score(criterion, example))

        score = sum(per_criterion[name] * weight for name, weight in weights.items())
        detail = ", ".join(f"{name}={value:.3f}" for name, value in per_criterion.items())
        return self._result(
            score,
            reason=f"weighted rubric score ({detail})",
            metadata={"criteria": per_criterion, "weights": weights},
        )
