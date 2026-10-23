"""Tests for the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval.cli import _parse_thresholds, main


def _write_dataset(path: Path) -> None:
    rows = [
        {"input": "q1", "expected": "a b c", "actual": "a b c", "context": ["a b c"]},
        {"input": "q2", "expected": "x y", "actual": "x y", "context": ["x y"]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_parse_thresholds_valid() -> None:
    assert _parse_thresholds("bleu=0.3,rouge_l=0.4") == {"bleu": 0.3, "rouge_l": 0.4}


def test_parse_thresholds_empty() -> None:
    assert _parse_thresholds("") == {}


def test_parse_thresholds_invalid_format() -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_thresholds("bleu")


def test_parse_thresholds_invalid_value() -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_thresholds("bleu=abc")


def test_list_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list-metrics"]) == 0
    out = capsys.readouterr().out
    assert "bleu" in out


def test_list_datasets(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_dataset(tmp_path / "sample.jsonl")
    assert main(["list-datasets", "--directory", str(tmp_path)]) == 0
    assert "sample.jsonl" in capsys.readouterr().out


def test_list_datasets_missing_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list-datasets", "--directory", str(tmp_path / "nope")]) == 2
    assert "directory not found" in capsys.readouterr().err


def test_run_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset_path = tmp_path / "data.jsonl"
    _write_dataset(dataset_path)
    store_path = tmp_path / "runs.sqlite3"
    code = main(
        [
            "run",
            str(dataset_path),
            "--metrics",
            "exact_match,contains",
            "--store",
            str(store_path),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"] == "data"
    assert payload["persisted"] is True
    assert store_path.exists()


def test_run_command_no_persist(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset_path = tmp_path / "data.jsonl"
    _write_dataset(dataset_path)
    code = main(
        [
            "run",
            str(dataset_path),
            "--metrics",
            "bleu",
            "--store",
            str(tmp_path / "runs.sqlite3"),
            "--no-persist",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["persisted"] is False


def test_run_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["run", str(path)])


def test_compare_and_gate_flow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset_path = tmp_path / "data.jsonl"
    _write_dataset(dataset_path)
    store_path = str(tmp_path / "runs.sqlite3")

    main(["run", str(dataset_path), "--metrics", "bleu", "--name", "baseline",
          "--store", store_path])
    baseline = json.loads(capsys.readouterr().out)["id"]

    main(["run", str(dataset_path), "--metrics", "bleu", "--name", "candidate",
          "--store", store_path])
    candidate = json.loads(capsys.readouterr().out)["id"]

    assert main(["compare", baseline, candidate, "--store", store_path]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert "counts" in comparison

    assert (
        main(["gate", baseline, "--metric-thresholds", "bleu=0.1", "--store", store_path])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["passed"] is True

    assert (
        main(["gate", baseline, "--metric-thresholds", "bleu=1.5", "--store", store_path])
        == 1
    )
    assert json.loads(capsys.readouterr().out)["passed"] is False


def test_gate_unknown_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["gate", "missing", "--store", str(tmp_path / "runs.sqlite3")])
    assert code == 2
    assert "unknown run" in capsys.readouterr().err


def test_compare_unknown_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["compare", "a", "b", "--store", str(tmp_path / "runs.sqlite3")])
    assert code == 2


def test_version_flag() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
