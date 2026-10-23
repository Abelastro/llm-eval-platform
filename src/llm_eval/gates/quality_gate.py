"""CI quality gate that enforces per-metric and overall thresholds."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from llm_eval.models import GateResult, Run


class QualityGate(BaseModel):
    """Threshold-based quality gate for evaluation runs.

    Attributes:
        metric_thresholds: Minimum mean score required per metric.
        overall_pass_rate: Minimum overall pass rate required.
        max_regressions: Maximum allowed regressions when a comparison is
            supplied to :meth:`evaluate_comparison`.
    """

    model_config = ConfigDict(extra="forbid")

    metric_thresholds: dict[str, float] = Field(default_factory=dict)
    overall_pass_rate: float = 0.0
    max_regressions: int = 0

    def evaluate(self, run: Run) -> GateResult:
        """Evaluate ``run`` against the configured thresholds.

        Args:
            run: The run to gate.

        Returns:
            A :class:`GateResult` describing pass/fail and the reasons.
        """

        reasons: list[str] = []
        observed: dict[str, float] = {}
        passed = True

        for metric, threshold in sorted(self.metric_thresholds.items()):
            score = run.summary.get(metric)
            if score is None:
                passed = False
                reasons.append(f"metric '{metric}' not present in run summary")
                continue
            observed[metric] = score
            if score < threshold:
                passed = False
                reasons.append(
                    f"metric '{metric}' score {score:.4f} below threshold {threshold:.4f}"
                )
            else:
                reasons.append(
                    f"metric '{metric}' score {score:.4f} meets threshold {threshold:.4f}"
                )

        observed_rate = run.pass_rate()
        if observed_rate < self.overall_pass_rate:
            passed = False
            reasons.append(
                f"overall pass rate {observed_rate:.4f} below floor "
                f"{self.overall_pass_rate:.4f}"
            )
        else:
            reasons.append(
                f"overall pass rate {observed_rate:.4f} meets floor "
                f"{self.overall_pass_rate:.4f}"
            )

        return GateResult(
            passed=passed,
            reasons=reasons,
            metrics=observed,
            overall_pass_rate=observed_rate,
        )

    def evaluate_comparison(self, comparison) -> GateResult:
        """Evaluate a comparison result against the regression budget.

        Args:
            comparison: A
                :class:`~llm_eval.runners.compare.ComparisonResult`.

        Returns:
            A :class:`GateResult` describing pass/fail and the reasons.
        """

        reasons: list[str] = []
        regressions = len(comparison.regressions)
        broken = len(comparison.broken)
        passed = regressions <= self.max_regressions
        reasons.append(
            f"{regressions} regression(s) allowed maximum {self.max_regressions}"
        )
        if broken:
            passed = False
            reasons.append(f"{broken} broken example(s) detected")
        return GateResult(
            passed=passed,
            reasons=reasons,
            metrics={
                "regressions": float(regressions),
                "broken": float(broken),
                "fixed": float(len(comparison.fixed)),
            },
            overall_pass_rate=0.0,
        )


def evaluate_gate(run: Run, metric_thresholds: dict[str, float],
                  overall_pass_rate: float = 0.0) -> GateResult:
    """Convenience function that builds and applies a :class:`QualityGate`.

    Args:
        run: The run to gate.
        metric_thresholds: Minimum mean score per metric.
        overall_pass_rate: Minimum overall pass rate.

    Returns:
        The resulting :class:`GateResult`.
    """

    gate = QualityGate(
        metric_thresholds=dict(metric_thresholds),
        overall_pass_rate=overall_pass_rate,
    )
    return gate.evaluate(run)
