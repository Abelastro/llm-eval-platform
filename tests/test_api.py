"""Tests for the FastAPI application."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_eval.api.app import create_app
from llm_eval.config import Settings
from llm_eval.datasets.registry import DatasetRegistry
from llm_eval.store.sqlite_store import SQLiteRunStore


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Return a test client backed by isolated registries and store.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        A :class:`TestClient` instance.
    """

    settings = Settings(store_path=str(tmp_path / "api.sqlite3"))
    app = create_app(
        settings=settings,
        dataset_registry=DatasetRegistry(),
        store=SQLiteRunStore(tmp_path / "api.sqlite3"),
    )
    return TestClient(app)


def _dataset_payload() -> dict:
    return {
        "name": "api-set",
        "version": "1",
        "examples": [
            {"input": "q1", "expected": "a b c", "actual": "a b c", "context": ["a b c"]},
            {"input": "q2", "expected": "x y", "actual": "x", "context": ["x y"]},
        ],
    }


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_metrics(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "bleu" in response.json()["metrics"]


def test_create_and_list_datasets(client: TestClient) -> None:
    created = client.post("/datasets", json=_dataset_payload())
    assert created.status_code == 201
    listed = client.get("/datasets").json()
    assert listed[0]["name"] == "api-set"
    assert listed[0]["num_examples"] == 2


def test_create_dataset_validation_error(client: TestClient) -> None:
    response = client.post("/datasets", json={"name": ""})
    assert response.status_code == 422


def test_create_run_from_registered_dataset(client: TestClient) -> None:
    client.post("/datasets", json=_dataset_payload())
    response = client.post(
        "/runs", json={"dataset": "api-set", "metrics": ["exact_match", "bleu"]}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["dataset"] == "api-set"
    assert "bleu" in body["summary"]


def test_create_run_inline_examples(client: TestClient) -> None:
    response = client.post(
        "/runs",
        json={
            "metrics": ["contains"],
            "examples": [{"expected": "a", "actual": "a b"}],
        },
    )
    assert response.status_code == 201


def test_create_run_unknown_metric(client: TestClient) -> None:
    response = client.post(
        "/runs",
        json={"metrics": ["nope"], "examples": [{"input": "x"}]},
    )
    assert response.status_code == 400


def test_create_run_unknown_dataset(client: TestClient) -> None:
    response = client.post("/runs", json={"dataset": "missing", "metrics": ["bleu"]})
    assert response.status_code == 404


def test_create_run_requires_dataset_or_examples(client: TestClient) -> None:
    response = client.post("/runs", json={"metrics": ["bleu"]})
    assert response.status_code == 400


def test_get_and_list_runs(client: TestClient) -> None:
    run = client.post(
        "/runs", json={"metrics": ["bleu"], "examples": [{"expected": "a", "actual": "a"}]}
    ).json()
    fetched = client.get(f"/runs/{run['id']}")
    assert fetched.status_code == 200
    listed = client.get("/runs").json()
    assert any(item["id"] == run["id"] for item in listed)


def test_get_unknown_run(client: TestClient) -> None:
    assert client.get("/runs/nope").status_code == 404


def test_compare_endpoint(client: TestClient) -> None:
    base = client.post(
        "/runs", json={"metrics": ["bleu"], "examples": [{"expected": "a b", "actual": "a b"}]}
    ).json()
    cand = client.post(
        "/runs", json={"metrics": ["bleu"], "examples": [{"expected": "a b", "actual": "x"}]}
    ).json()
    response = client.post(
        "/compare", json={"baseline_id": base["id"], "candidate_id": cand["id"]}
    )
    assert response.status_code == 200
    assert "counts" in response.json()


def test_compare_unknown_run(client: TestClient) -> None:
    response = client.post("/compare", json={"baseline_id": "a", "candidate_id": "b"})
    assert response.status_code == 404


def test_gate_endpoint(client: TestClient) -> None:
    run = client.post(
        "/runs", json={"metrics": ["bleu"], "examples": [{"expected": "a b c", "actual": "a b c"}]}
    ).json()
    response = client.post(
        "/gates/evaluate",
        json={"run_id": run["id"], "metric_thresholds": {"bleu": 0.1}},
    )
    assert response.status_code == 200
    assert response.json()["passed"] is True


def test_gate_endpoint_failure(client: TestClient) -> None:
    run = client.post(
        "/runs", json={"metrics": ["bleu"], "examples": [{"expected": "a b c", "actual": "z"}]}
    ).json()
    response = client.post(
        "/gates/evaluate",
        json={"run_id": run["id"], "metric_thresholds": {"bleu": 0.99}},
    )
    assert response.json()["passed"] is False


def test_gate_unknown_run(client: TestClient) -> None:
    response = client.post("/gates/evaluate", json={"run_id": "nope"})
    assert response.status_code == 404
