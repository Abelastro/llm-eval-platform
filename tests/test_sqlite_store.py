"""Tests for the SQLite run store."""

from __future__ import annotations

import pytest

from llm_eval.models import MetricResult, Run
from llm_eval.store.sqlite_store import SQLiteRunStore


def _run(run_id: str = "run-1") -> Run:
    return Run(
        id=run_id,
        dataset="d",
        name="nightly",
        results={
            0: {
                "bleu": MetricResult(
                    name="bleu", score=0.5, passed=True, reason="ok", metadata={"k": 1}
                )
            },
            1: {
                "bleu": MetricResult(name="bleu", score=0.2, passed=False),
                "rouge_l": MetricResult(name="rouge_l", score=0.55, passed=True),
            },
        },
        summary={"bleu": 0.35, "rouge_l": 0.55},
    )


def test_save_and_get_roundtrip(store: SQLiteRunStore) -> None:
    run = _run()
    store.save_run(run)
    loaded = store.get_run(run.id)
    assert loaded.dataset == "d"
    assert loaded.name == "nightly"
    assert loaded.summary == run.summary
    assert loaded.results[0]["bleu"].score == 0.5
    assert loaded.results[0]["bleu"].metadata == {"k": 1}
    assert loaded.results[1]["bleu"].passed is False


def test_get_unknown_raises(store: SQLiteRunStore) -> None:
    with pytest.raises(KeyError):
        store.get_run("nope")


def test_list_runs_empty(store: SQLiteRunStore) -> None:
    assert store.list_runs() == []


def test_list_runs_returns_all(store: SQLiteRunStore) -> None:
    store.save_run(_run("a"))
    store.save_run(_run("b"))
    ids = {run.id for run in store.list_runs()}
    assert ids == {"a", "b"}


def test_list_runs_limit(store: SQLiteRunStore) -> None:
    store.save_run(_run("a"))
    store.save_run(_run("b"))
    assert len(store.list_runs(limit=1)) == 1


def test_count(store: SQLiteRunStore) -> None:
    assert store.count() == 0
    store.save_run(_run("a"))
    assert store.count() == 1


def test_save_is_idempotent(store: SQLiteRunStore) -> None:
    run = _run("a")
    store.save_run(run)
    run.summary["bleu"] = 0.99
    store.save_run(run)
    assert store.count() == 1
    assert store.get_run("a").summary["bleu"] == 0.99


def test_delete_run(store: SQLiteRunStore) -> None:
    store.save_run(_run("a"))
    store.delete_run("a")
    assert store.count() == 0


def test_clear(store: SQLiteRunStore) -> None:
    store.save_run(_run("a"))
    store.save_run(_run("b"))
    store.clear()
    assert store.count() == 0


def test_schema_created_at_path(tmp_path) -> None:
    store = SQLiteRunStore(tmp_path / "nested" / "runs.sqlite3")
    store.save_run(_run("a"))
    assert store.count() == 1
