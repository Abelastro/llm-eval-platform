"""Pydantic domain models shared across the evaluation platform."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    """Return the current timezone-aware UTC timestamp.

    Returns:
        A timezone-aware :class:`datetime` in UTC.
    """

    return datetime.now(timezone.utc)


class Example(BaseModel):
    """A single evaluation case.

    Attributes:
        input: The prompt or question presented to the system under test.
        expected: The reference or gold answer, when available.
        actual: The model/system output being scored.
        context: Retrieved context supporting a RAG answer, if any.
        metadata: Arbitrary user-supplied annotations.
    """

    model_config = ConfigDict(extra="forbid")

    input: str = ""
    expected: str | None = None
    actual: str = ""
    context: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    """A versioned collection of evaluation examples.

    Attributes:
        name: Logical dataset name used as a registry key.
        version: Monotonic version string (e.g. ``"1"`` or ``"v2"``).
        examples: The ordered list of examples.
        metadata: Arbitrary dataset-level annotations.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "1"
    examples: list[Example] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        """Reject blank dataset names.

        Args:
            value: Candidate dataset name.

        Returns:
            The validated name.

        Raises:
            ValueError: If the name is empty or whitespace only.
        """

        if not value or not value.strip():
            raise ValueError("dataset name must not be blank")
        return value


class MetricResult(BaseModel):
    """The outcome of scoring one example with one metric.

    Attributes:
        name: Metric identifier that produced the result.
        score: Normalized score in the metric's natural range.
        passed: Whether the score satisfied the metric's pass threshold.
        reason: Human-readable explanation of the score.
        metadata: Optional diagnostic payload.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    score: float
    passed: bool
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Run(BaseModel):
    """A single execution of a dataset through a set of metrics.

    Attributes:
        id: Unique run identifier; generated when omitted.
        dataset: Name of the dataset that was evaluated.
        name: Human-readable run label.
        created_at: Timezone-aware creation timestamp.
        results: Per-example metric results keyed by example index.
        summary: Aggregate statistics keyed by metric name.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    dataset: str
    name: str = "run"
    created_at: datetime = Field(default_factory=_utcnow)
    results: dict[int, dict[str, MetricResult]] = Field(default_factory=dict)
    summary: dict[str, float] = Field(default_factory=dict)

    def pass_rate(self, metric: str | None = None) -> float:
        """Compute the pass rate across examples.

        Args:
            metric: Restrict the calculation to one metric name. When ``None``
                every metric result contributes.

        Returns:
            The fraction of passing results, or ``0.0`` when no results exist.
        """

        total = 0
        passed = 0
        for per_example in self.results.values():
            for name, result in per_example.items():
                if metric is not None and name != metric:
                    continue
                total += 1
                if result.passed:
                    passed += 1
        if total == 0:
            return 0.0
        return passed / total


class GateResult(BaseModel):
    """The verdict produced by a quality gate evaluation.

    Attributes:
        passed: Whether every configured threshold was satisfied.
        reasons: Human-readable explanations for the verdict.
        metrics: Per-metric observed scores that were checked.
        overall_pass_rate: Observed overall pass rate across the run.
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    overall_pass_rate: float = 0.0
