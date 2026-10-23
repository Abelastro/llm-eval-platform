"""Tests for dataset loaders and registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval.datasets.loaders import (
    load_dataset_from_csv,
    load_dataset_from_jsonl,
    load_dataset_from_memory,
)
from llm_eval.datasets.registry import DatasetRegistry
from llm_eval.models import Dataset, Example


def _write_jsonl(path: Path) -> None:
    rows = [
        {
            "input": "q1",
            "expected": "a1",
            "actual": "a1",
            "context": ["ctx one", "ctx two"],
            "metadata": {"k": 1},
        },
        {"input": "q2", "expected": "a2", "actual": "wrong", "context": "single ctx"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_load_jsonl_basic(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    _write_jsonl(path)
    dataset = load_dataset_from_jsonl(path)
    assert dataset.name == "data"
    assert len(dataset.examples) == 2
    assert dataset.examples[0].context == ["ctx one", "ctx two"]
    assert dataset.examples[0].metadata == {"k": 1}


def test_load_jsonl_context_string(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    _write_jsonl(path)
    dataset = load_dataset_from_jsonl(path)
    assert dataset.examples[1].context == ["single ctx"]


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"input": "q"}\n\n\n', encoding="utf-8")
    dataset = load_dataset_from_jsonl(path)
    assert len(dataset.examples) == 1


def test_load_jsonl_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset_from_jsonl(path)


def test_load_jsonl_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset_from_jsonl(tmp_path / "missing.jsonl")


def test_load_csv(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text(
        "input,expected,actual,context,metadata\n"
        'q1,a1,a1,"ctx one||ctx two","{""source"": ""manual""}"\n',
        encoding="utf-8",
    )
    dataset = load_dataset_from_csv(path)
    assert dataset.examples[0].context == ["ctx one", "ctx two"]
    assert dataset.examples[0].metadata == {"source": "manual"}


def test_load_csv_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset_from_csv(tmp_path / "missing.csv")


def test_load_from_memory() -> None:
    dataset = load_dataset_from_memory("mem", [{"input": "q", "expected": "a"}])
    assert dataset.name == "mem"
    assert dataset.examples[0].input == "q"


def test_load_from_memory_blank_expected_becomes_none() -> None:
    dataset = load_dataset_from_memory("mem", [{"input": "q", "expected": ""}])
    assert dataset.examples[0].expected is None


def test_dataset_registry_register_and_get(dataset_registry: DatasetRegistry) -> None:
    dataset = Dataset(name="d", version="1", examples=[Example(input="x")])
    dataset_registry.register(dataset)
    assert dataset_registry.get("d").version == "1"


def test_dataset_registry_latest_version(dataset_registry: DatasetRegistry) -> None:
    dataset_registry.register(Dataset(name="d", version="1"))
    dataset_registry.register(Dataset(name="d", version="2"))
    assert dataset_registry.get("d").version == "2"


def test_dataset_registry_resolve_reference(dataset_registry: DatasetRegistry) -> None:
    dataset_registry.register(Dataset(name="d", version="1"))
    dataset_registry.register(Dataset(name="d", version="2"))
    assert dataset_registry.resolve("d@1").version == "1"


def test_dataset_registry_unknown_raises(dataset_registry: DatasetRegistry) -> None:
    with pytest.raises(KeyError):
        dataset_registry.get("missing")


def test_dataset_registry_contains(dataset_registry: DatasetRegistry) -> None:
    dataset_registry.register(Dataset(name="d"))
    assert "d" in dataset_registry
    assert "other" not in dataset_registry
    assert 123 not in dataset_registry


def test_dataset_registry_len_and_versions(dataset_registry: DatasetRegistry) -> None:
    dataset_registry.register(Dataset(name="d", version="1"))
    dataset_registry.register(Dataset(name="d", version="2"))
    assert len(dataset_registry) == 2
    assert dataset_registry.list_versions("d") == ["1", "2"]
    assert dataset_registry.list_names() == ["d"]
