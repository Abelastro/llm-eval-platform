"""Loaders that turn serialized data into :class:`~llm_eval.models.Dataset`."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from llm_eval.models import Dataset, Example

_CONTEXT_SEPARATOR = "||"


def _coerce_context(value: Any) -> list[str]:
    """Normalize a context value into a list of non-empty strings.

    Args:
        value: A list, a ``||``-delimited string, ``None``, or a scalar.

    Returns:
        A list of trimmed context strings.
    """

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    text = str(value)
    if not text.strip():
        return []
    if _CONTEXT_SEPARATOR in text:
        return [part.strip() for part in text.split(_CONTEXT_SEPARATOR) if part.strip()]
    return [text.strip()]


def _example_from_mapping(row: Mapping[str, Any]) -> Example:
    """Build an :class:`Example` from a loosely-typed mapping.

    Args:
        row: Mapping that may contain ``input``, ``expected``, ``actual``,
            ``context``, and ``metadata`` keys.

    Returns:
        A validated example.
    """

    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata) if metadata.strip() else {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    return Example(
        input=str(row.get("input") or ""),
        expected=(None if row.get("expected") in (None, "") else str(row["expected"])),
        actual=str(row.get("actual") or ""),
        context=_coerce_context(row.get("context")),
        metadata=dict(metadata),
    )


def load_dataset_from_memory(
    name: str,
    rows: Iterable[Mapping[str, Any]],
    version: str = "1",
    metadata: Mapping[str, Any] | None = None,
) -> Dataset:
    """Build a dataset from an in-memory iterable of mappings.

    Args:
        name: Dataset name.
        rows: Iterable of mapping rows.
        version: Dataset version string.
        metadata: Optional dataset-level metadata.

    Returns:
        A populated :class:`Dataset`.
    """

    examples = [_example_from_mapping(row) for row in rows]
    return Dataset(
        name=name,
        version=version,
        examples=examples,
        metadata=dict(metadata or {}),
    )


def load_dataset_from_jsonl(
    path: str | Path,
    name: str | None = None,
    version: str = "1",
    metadata: Mapping[str, Any] | None = None,
) -> Dataset:
    """Load a dataset from a JSON Lines file.

    Args:
        path: Path to the ``.jsonl`` file.
        name: Dataset name; defaults to the file stem.
        version: Dataset version string.
        metadata: Optional dataset-level metadata.

    Returns:
        A populated :class:`Dataset`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line is not a JSON object.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"dataset file not found: {file_path}")

    rows: list[Mapping[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"line {line_number} must be a JSON object, got {type(payload).__name__}"
                )
            rows.append(payload)

    return load_dataset_from_memory(
        name=name or file_path.stem,
        rows=rows,
        version=version,
        metadata=metadata,
    )


def load_dataset_from_csv(
    path: str | Path,
    name: str | None = None,
    version: str = "1",
    metadata: Mapping[str, Any] | None = None,
) -> Dataset:
    """Load a dataset from a CSV file.

    The CSV is expected to use ``input``, ``expected``, ``actual``, ``context``,
    and ``metadata`` column headers. Multiple context passages are delimited by
    ``||`` and ``metadata`` is parsed as a JSON object when present.

    Args:
        path: Path to the ``.csv`` file.
        name: Dataset name; defaults to the file stem.
        version: Dataset version string.
        metadata: Optional dataset-level metadata.

    Returns:
        A populated :class:`Dataset`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"dataset file not found: {file_path}")

    rows: list[Mapping[str, Any]] = []
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: value for key, value in row.items() if key is not None})

    return load_dataset_from_memory(
        name=name or file_path.stem,
        rows=rows,
        version=version,
        metadata=metadata,
    )
