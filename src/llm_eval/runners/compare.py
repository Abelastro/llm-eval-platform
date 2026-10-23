"""Compare two runs and classify per-metric changes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_eval.models import Run

Status = Literal["improvement", "regression", "fixed", "broken", "unchanged"]

_EPSILON = 1e-6


class MetricDelta(BaseModel):
    """A single per-example, per-metric comparison record.

    Attributes:
        example_index: Index of the example within the dataset.
        metric: Metric name.
        baseline_score: Score in the baseline run.
        candidate_score: Score in the candidate run.
        delta: ``candidate_score - baseline_score``.
        baseline_passed: Pass flag in the baseline run.
        candidate_passed: Pass flag in the candidate run.
        status: Classification of the change.
    """

    model_config = ConfigDict(extra="forbid")

    example_index: int
    metric: str
    baseline_score: float
    candidate_score: float
    delta: float
    baseline_passed: bool
    candidate_passed: bool
    status: Status


class ComparisonResult(BaseModel):
    """Aggregate comparison of a baseline and candidate run.

    Attributes:
        baseline_id: Identifier of the baseline run.
        candidate_id: Identifier of the candidate run.
        deltas: Every classified per-metric change.
        regressions: Deltas whose status is ``regression``.
        improvements: Deltas whose status is ``improvement``.
        fixed: Deltas whose status is ``fixed``.
        broken: Deltas whose status is ``broken``.
        scores: Mean score per metric for both runs.
    """

    model_config = ConfigDict(extra="forbid")

    baseline_id: str
    candidate_id: str
    deltas: list[MetricDelta] = Field(default_factory=list)
    regressions: list[MetricDelta] = Field(default_factory=list)
    improvements: list[MetricDelta] = Field(default_factory=list)
    fixed: list[MetricDelta] = Field(default_factory=list)
    broken: list[MetricDelta] = Field(default_factory=list)
    scores: dict[str, dict[str, float]] = Field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        """Return the size of every change category.

        Returns:
            Mapping of category name to count.
        """

        return {
            "regressions": len(self.regressions),
            "improvements": len(self.improvements),
            "fixed": len(self.fixed),
            "broken": len(self.broken),
        }

    def has_regressions(self) -> bool:
        """Return whether any metric regressed.

        Returns:
            ``True`` when at least one regression or break was detected.
        """

        return bool(self.regressions or self.broken)


def _classify(baseline_passed: bool, candidate_passed: bool, delta: float) -> Status:
    """Classify a single metric change.

    Args:
        baseline_passed: Whether the baseline result passed.
        candidate_passed: Whether the candidate result passed.
        delta: Score delta.

    Returns:
        The change status.
    """

    if not baseline_passed and candidate_passed:
        return "fixed"
    if baseline_passed and not candidate_passed:
        return "broken"
    if delta > _EPSILON:
        return "improvement"
    if delta < -_EPSILON:
        return "regression"
    return "unchanged"


def _score_delta(baseline: float, candidate: float) -> float:
    """Return a numerically stable score delta.

    Args:
        baseline: Baseline mean score.
        candidate: Candidate mean score.

    Returns:
        The difference rounded to ten decimal places.
    """

    return round(candidate - baseline, 10)


def compare_runs(baseline: Run, candidate: Run) -> ComparisonResult:
    """Compare a baseline run against a candidate run.

    Args:
        baseline: The reference run.
        candidate: The run to compare against the baseline.

    Returns:
        A :class:`ComparisonResult` describing every change.
    """

    deltas: list[MetricDelta] = []
    for index in sorted(set(baseline.results) | set(candidate.results)):
        base_example = baseline.results.get(index, {})
        cand_example = candidate.results.get(index, {})
        for metric in sorted(set(base_example) | set(cand_example)):
            base_result = base_example.get(metric)
            cand_result = cand_example.get(metric)
            baseline_score = base_result.score if base_result else 0.0
            candidate_score = cand_result.score if cand_result else 0.0
            baseline_passed = base_result.passed if base_result else False
            candidate_passed = cand_result.passed if cand_result else False
            delta = candidate_score - baseline_score
            deltas.append(
                MetricDelta(
                    example_index=index,
                    metric=metric,
                    baseline_score=baseline_score,
                    candidate_score=candidate_score,
                    delta=delta,
                    baseline_passed=baseline_passed,
                    candidate_passed=candidate_passed,
                    status=_classify(baseline_passed, candidate_passed, delta),
                )
            )

    metrics = sorted(set(baseline.summary) | set(candidate.summary))
    scores = {
        metric: {
            "baseline": baseline.summary.get(metric, 0.0),
            "candidate": candidate.summary.get(metric, 0.0),
            "delta": _score_delta(
                baseline.summary.get(metric, 0.0),
                candidate.summary.get(metric, 0.0),
            ),
        }
        for metric in metrics
    }

    return ComparisonResult(
        baseline_id=baseline.id,
        candidate_id=candidate.id,
        deltas=deltas,
        regressions=[
            delta for delta in deltas if delta.status in ("regression", "broken")
        ],
        improvements=[
            delta for delta in deltas if delta.status in ("improvement", "fixed")
        ],
        fixed=[delta for delta in deltas if delta.status == "fixed"],
        broken=[delta for delta in deltas if delta.status == "broken"],
        scores=scores,
    )
