"""SQLite-backed persistence for evaluation runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from llm_eval.models import MetricResult, Run

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    summary TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL,
    example_index INTEGER NOT NULL,
    metric TEXT NOT NULL,
    score REAL NOT NULL,
    passed INTEGER NOT NULL,
    reason TEXT NOT NULL,
    metadata TEXT NOT NULL,
    PRIMARY KEY (run_id, example_index, metric),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
"""


class SQLiteRunStore:
    """Persist and retrieve :class:`~llm_eval.models.Run` objects.

    Each operation opens a short-lived connection so the store is safe to use
    from multiple threads (for example behind the FastAPI test client).
    """

    def __init__(self, path: str | Path = "llm_eval_runs.sqlite3") -> None:
        """Initialize the store.

        Args:
            path: Filesystem path of the SQLite database.
        """

        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection configured for foreign keys.

        Returns:
            A SQLite connection.
        """

        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        """Create the schema if it does not already exist."""

        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def save_run(self, run: Run) -> Run:
        """Persist a run and all of its metric results.

        Args:
            run: The run to persist.

        Returns:
            The persisted run.
        """

        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs (id, dataset, name, created_at, summary) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run.id,
                    run.dataset,
                    run.name,
                    run.created_at.isoformat(),
                    json.dumps(run.summary),
                ),
            )
            connection.execute("DELETE FROM results WHERE run_id = ?", (run.id,))
            rows = []
            for index, per_example in run.results.items():
                for metric, result in per_example.items():
                    rows.append(
                        (
                            run.id,
                            int(index),
                            metric,
                            float(result.score),
                            int(result.passed),
                            result.reason,
                            json.dumps(result.metadata),
                        )
                    )
            connection.executemany(
                "INSERT INTO results "
                "(run_id, example_index, metric, score, passed, reason, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return run

    def get_run(self, run_id: str) -> Run:
        """Retrieve a run by identifier.

        Args:
            run_id: The run identifier.

        Returns:
            The stored run with its results.

        Raises:
            KeyError: If no run matches ``run_id``.
        """

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown run: {run_id}")
            result_rows = connection.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY example_index, metric",
                (run_id,),
            ).fetchall()

        results: dict[int, dict[str, MetricResult]] = {}
        for result_row in result_rows:
            index = int(result_row["example_index"])
            results.setdefault(index, {})[result_row["metric"]] = MetricResult(
                name=result_row["metric"],
                score=float(result_row["score"]),
                passed=bool(result_row["passed"]),
                reason=result_row["reason"],
                metadata=json.loads(result_row["metadata"]),
            )

        return Run(
            id=row["id"],
            dataset=row["dataset"],
            name=row["name"],
            created_at=datetime.fromisoformat(row["created_at"]),
            results=results,
            summary=json.loads(row["summary"]),
        )

    def list_runs(self, limit: int | None = None) -> list[Run]:
        """List stored runs, newest first.

        Args:
            limit: Optional maximum number of runs to return.

        Returns:
            A list of stored runs.
        """

        query = "SELECT id FROM runs ORDER BY created_at DESC, id DESC"
        params: tuple[object, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (int(limit),)
        with self._connect() as connection:
            ids = [row["id"] for row in connection.execute(query, params).fetchall()]
        return [self.get_run(run_id) for run_id in ids]

    def delete_run(self, run_id: str) -> None:
        """Delete a run and its results.

        Args:
            run_id: The run identifier.
        """

        with self._connect() as connection:
            connection.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
            connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    def count(self) -> int:
        """Return the number of stored runs.

        Returns:
            The run count.
        """

        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
        return int(row["n"])

    def clear(self) -> None:
        """Delete every stored run and result."""

        with self._connect() as connection:
            connection.execute("DELETE FROM results")
            connection.execute("DELETE FROM runs")
