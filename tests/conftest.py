"""Shared pytest fixtures for the llm-eval-platform test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval.datasets.registry import DatasetRegistry
from llm_eval.models import Dataset, Example
from llm_eval.store.sqlite_store import SQLiteRunStore


@pytest.fixture(autouse=True)
def _clean_dataset_registry():
    """Ensure the global dataset registry is empty around each test."""

    from llm_eval.datasets.registry import get_registry

    registry = get_registry()
    registry.clear()
    yield
    registry.clear()


@pytest.fixture
def sample_examples() -> list[Example]:
    """Return a small, deterministic set of examples.

    Returns:
        A list of example objects.
    """

    return [
        Example(
            input="What is the capital of France?",
            expected="Paris is the capital of France.",
            actual="The capital of France is Paris.",
            context=["France is a country in Europe. Its capital is Paris."],
        ),
        Example(
            input="Who wrote Hamlet?",
            expected="William Shakespeare wrote Hamlet.",
            actual="Hamlet was written by William Shakespeare.",
            context=["Hamlet is a tragedy written by William Shakespeare."],
        ),
        Example(
            input="What color is the sky?",
            expected="The sky is blue.",
            actual="Bananas are yellow.",
            context=["Seventeen apples ripen in autumn weather."],
        ),
    ]


@pytest.fixture
def sample_dataset(sample_examples: list[Example]) -> Dataset:
    """Return a dataset wrapping the sample examples.

    Args:
        sample_examples: The example fixture.

    Returns:
        A dataset named ``sample``.
    """

    return Dataset(name="sample", version="1", examples=sample_examples)


@pytest.fixture
def store(tmp_path: Path) -> SQLiteRunStore:
    """Return a temporary SQLite run store.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        A fresh store backed by a temporary file.
    """

    return SQLiteRunStore(tmp_path / "runs.sqlite3")


@pytest.fixture
def dataset_registry() -> DatasetRegistry:
    """Return a fresh dataset registry.

    Returns:
        An empty :class:`DatasetRegistry`.
    """

    return DatasetRegistry()
