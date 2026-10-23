"""Runnable example: evaluate a sample dataset, compare runs, and gate CI.

Run from the repository root with::

    PYTHONPATH=src python3 examples/run_example.py

The script is fully offline and deterministic.
"""

from __future__ import annotations

from pathlib import Path

from llm_eval.datasets.loaders import load_dataset_from_jsonl
from llm_eval.gates.quality_gate import QualityGate
from llm_eval.models import Dataset, Example
from llm_eval.runners.compare import compare_runs
from llm_eval.runners.runner import Runner

EXAMPLES_DIR = Path(__file__).resolve().parent
METRICS = ["exact_match", "contains", "bleu", "rouge_l", "faithfulness", "relevance"]


def main() -> None:
    """Load a dataset, run two versions, compare, and evaluate a gate."""

    dataset = load_dataset_from_jsonl(EXAMPLES_DIR / "sample_qa.jsonl")
    print(f"Loaded dataset '{dataset.name}' with {len(dataset.examples)} examples")

    runner = Runner.from_names(METRICS)
    baseline = runner.run(dataset, name="baseline")
    print("Baseline summary:")
    for metric, score in sorted(baseline.summary.items()):
        print(f"  {metric:<16} {score:.4f}")

    degraded_examples = [
        Example(
            input=example.input,
            expected=example.expected,
            actual=(example.actual[: max(1, len(example.actual) // 2)] if index % 2 else ""),
            context=example.context,
            metadata=example.metadata,
        )
        for index, example in enumerate(dataset.examples)
    ]
    candidate = runner.run(
        Dataset(name="sample_qa", version="2", examples=degraded_examples),
        name="candidate",
    )

    comparison = compare_runs(baseline, candidate)
    print("Comparison counts:", comparison.counts())

    gate = QualityGate(
        metric_thresholds={"bleu": 0.05, "rouge_l": 0.1},
        overall_pass_rate=0.5,
    )
    gate_result = gate.evaluate(candidate)
    print("Gate passed:", gate_result.passed)
    for reason in gate_result.reasons:
        print("  -", reason)


if __name__ == "__main__":
    main()
