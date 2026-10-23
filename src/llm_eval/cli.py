"""Command-line interface for the evaluation platform."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from llm_eval import __version__
from llm_eval.config import get_settings
from llm_eval.datasets.loaders import load_dataset_from_csv, load_dataset_from_jsonl
from llm_eval.gates.quality_gate import QualityGate
from llm_eval.metrics.registry import get_registry as get_metric_registry
from llm_eval.models import Dataset
from llm_eval.runners.compare import compare_runs
from llm_eval.runners.runner import Runner
from llm_eval.store.sqlite_store import SQLiteRunStore

DEFAULT_METRICS = ("exact_match", "contains", "bleu", "rouge_l")


def _parse_thresholds(value: str) -> dict[str, float]:
    """Parse a ``name=value,name=value`` threshold specification.

    Args:
        value: The raw CLI argument.

    Returns:
        A mapping of metric name to threshold.

    Raises:
        argparse.ArgumentTypeError: If the specification is malformed.
    """

    thresholds: dict[str, float] = {}
    if not value.strip():
        return thresholds
    for chunk in value.split(","):
        if "=" not in chunk:
            raise argparse.ArgumentTypeError(
                f"invalid threshold '{chunk}', expected name=value"
            )
        name, raw = chunk.split("=", 1)
        try:
            thresholds[name.strip()] = float(raw)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid threshold value for '{name.strip()}': {raw}"
            ) from error
    return thresholds


def _load_dataset(path: Path, name: str | None) -> Dataset:
    """Load a dataset from a JSONL or CSV file.

    Args:
        path: Path to the dataset file.
        name: Optional dataset name override.

    Returns:
        The loaded dataset.

    Raises:
        SystemExit: If the file extension is unsupported or missing.
    """

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_dataset_from_jsonl(path, name=name)
    if suffix == ".csv":
        return load_dataset_from_csv(path, name=name)
    raise SystemExit(f"unsupported dataset extension '{suffix}' (expected .jsonl or .csv)")


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the ``run`` subcommand.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A process exit code.
    """

    dataset = _load_dataset(Path(args.dataset), args.name)
    registry = get_metric_registry()
    metric_names = [name for name in args.metrics.split(",") if name.strip()]
    runner = Runner.from_names(metric_names, registry=registry)
    run = runner.run(dataset, name=args.name or dataset.name)

    if args.persist:
        SQLiteRunStore(args.store).save_run(run)

    payload = {
        "id": run.id,
        "dataset": run.dataset,
        "name": run.name,
        "summary": run.summary,
        "persisted": bool(args.persist),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Execute the ``compare`` subcommand.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A process exit code.
    """

    store = SQLiteRunStore(args.store)
    try:
        baseline = store.get_run(args.baseline_id)
        candidate = store.get_run(args.candidate_id)
    except KeyError as error:
        print(str(error), file=sys.stderr)
        return 2

    comparison = compare_runs(baseline, candidate)
    payload = {
        "baseline_id": comparison.baseline_id,
        "candidate_id": comparison.candidate_id,
        "counts": comparison.counts(),
        "scores": comparison.scores,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    """Execute the ``gate`` subcommand.

    Args:
        args: Parsed CLI arguments.

    Returns:
        ``0`` when the gate passes, ``1`` when it fails, ``2`` on error.
    """

    store = SQLiteRunStore(args.store)
    try:
        run = store.get_run(args.run_id)
    except KeyError as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        thresholds = _parse_thresholds(args.metric_thresholds)
    except argparse.ArgumentTypeError as error:
        print(str(error), file=sys.stderr)
        return 2

    gate = QualityGate(
        metric_thresholds=thresholds,
        overall_pass_rate=args.overall_pass_rate,
    )
    result = gate.evaluate(run)
    payload = {
        "passed": result.passed,
        "reasons": result.reasons,
        "metrics": result.metrics,
        "overall_pass_rate": result.overall_pass_rate,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.passed else 1


def _cmd_list_metrics(_: argparse.Namespace) -> int:
    """Execute the ``list-metrics`` subcommand.

    Args:
        _: Parsed CLI arguments (unused).

    Returns:
        A process exit code.
    """

    for name in get_metric_registry().list():
        print(name)
    return 0


def _cmd_list_datasets(args: argparse.Namespace) -> int:
    """Execute the ``list-datasets`` subcommand.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A process exit code.
    """

    directory = Path(args.directory)
    if not directory.exists():
        print(f"directory not found: {directory}", file=sys.stderr)
        return 2
    found = sorted(
        path.name
        for path in directory.iterdir()
        if path.suffix.lower() in {".jsonl", ".csv"}
    )
    for name in found:
        print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        The configured parser.
    """

    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="llm_eval",
        description="LLM/GenAI evaluation, regression, and CI quality gating.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="evaluate a dataset file")
    run_parser.add_argument("dataset", help="path to a .jsonl or .csv dataset")
    run_parser.add_argument("--name", default=None, help="run/dataset label")
    run_parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="comma-separated metric names",
    )
    run_parser.add_argument("--store", default=settings.store_path, help="SQLite store path")
    run_parser.add_argument(
        "--no-persist", dest="persist", action="store_false", help="skip persistence"
    )
    run_parser.set_defaults(func=_cmd_run, persist=True)

    compare_parser = subparsers.add_parser("compare", help="compare two stored runs")
    compare_parser.add_argument("baseline_id", help="baseline run id")
    compare_parser.add_argument("candidate_id", help="candidate run id")
    compare_parser.add_argument("--store", default=settings.store_path, help="SQLite store path")
    compare_parser.set_defaults(func=_cmd_compare)

    gate_parser = subparsers.add_parser("gate", help="evaluate a run against thresholds")
    gate_parser.add_argument("run_id", help="run id to gate")
    gate_parser.add_argument(
        "--metric-thresholds",
        default="",
        help="comma-separated name=value thresholds",
    )
    gate_parser.add_argument(
        "--overall-pass-rate", type=float, default=0.0, help="minimum overall pass rate"
    )
    gate_parser.add_argument("--store", default=settings.store_path, help="SQLite store path")
    gate_parser.set_defaults(func=_cmd_gate)

    list_metrics_parser = subparsers.add_parser(
        "list-metrics", help="list registered metrics"
    )
    list_metrics_parser.set_defaults(func=_cmd_list_metrics)

    list_datasets_parser = subparsers.add_parser(
        "list-datasets", help="list dataset files in a directory"
    )
    list_datasets_parser.add_argument(
        "--directory", default=settings.default_dataset_dir, help="directory to scan"
    )
    list_datasets_parser.set_defaults(func=_cmd_list_datasets)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
