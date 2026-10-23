"""FastAPI application exposing the evaluation platform over HTTP."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from llm_eval import __version__
from llm_eval.config import Settings, get_settings
from llm_eval.datasets.registry import DatasetRegistry, get_registry as get_dataset_registry
from llm_eval.gates.quality_gate import QualityGate
from llm_eval.metrics.registry import MetricRegistry, get_registry as get_metric_registry
from llm_eval.models import Dataset, Example, GateResult, Run
from llm_eval.runners.compare import compare_runs
from llm_eval.runners.runner import Runner
from llm_eval.store.sqlite_store import SQLiteRunStore


class CreateDatasetRequest(BaseModel):
    """Payload for registering a dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = "1"
    examples: list[Example] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateRunRequest(BaseModel):
    """Payload for executing an evaluation run."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[str]
    dataset: str | None = None
    dataset_version: str | None = None
    examples: list[Example] | None = None
    name: str = "run"
    persist: bool = True


class CompareRequest(BaseModel):
    """Payload for comparing two stored runs."""

    model_config = ConfigDict(extra="forbid")

    baseline_id: str
    candidate_id: str


class GateRequest(BaseModel):
    """Payload for evaluating a stored run against a quality gate."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    metric_thresholds: dict[str, float] = Field(default_factory=dict)
    overall_pass_rate: float = 0.0


class DatasetSummary(BaseModel):
    """Compact dataset listing entry."""

    name: str
    version: str
    num_examples: int


def create_app(
    settings: Settings | None = None,
    dataset_registry: DatasetRegistry | None = None,
    metric_registry: MetricRegistry | None = None,
    store: SQLiteRunStore | None = None,
) -> FastAPI:
    """Create a configured FastAPI application.

    Args:
        settings: Optional settings override.
        dataset_registry: Optional dataset registry override.
        metric_registry: Optional metric registry override.
        store: Optional run store override.

    Returns:
        The configured :class:`FastAPI` instance.
    """

    active_settings = settings or get_settings()
    datasets = dataset_registry or get_dataset_registry()
    metrics = metric_registry or get_metric_registry()
    run_store = store or SQLiteRunStore(active_settings.store_path)

    app = FastAPI(
        title=active_settings.app_name,
        version=__version__,
        description="LLM/GenAI evaluation, regression, and CI quality gating.",
    )
    app.state.settings = active_settings
    app.state.datasets = datasets
    app.state.metrics = metrics
    app.state.store = run_store

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return service liveness metadata.

        Returns:
            A small status document.
        """

        return {"status": "ok", "app": active_settings.app_name, "version": __version__}

    @app.get("/datasets", response_model=list[DatasetSummary])
    def list_datasets() -> list[DatasetSummary]:
        """List registered datasets.

        Returns:
            One summary per registered dataset version.
        """

        summaries: list[DatasetSummary] = []
        for name in datasets.list_names():
            for version in datasets.list_versions(name):
                dataset = datasets.get(name, version)
                summaries.append(
                    DatasetSummary(
                        name=dataset.name,
                        version=dataset.version,
                        num_examples=len(dataset.examples),
                    )
                )
        return summaries

    @app.post("/datasets", response_model=Dataset, status_code=201)
    def create_dataset(request: CreateDatasetRequest) -> Dataset:
        """Register a dataset.

        Args:
            request: Dataset registration payload.

        Returns:
            The registered dataset.
        """

        dataset = Dataset(
            name=request.name,
            version=request.version,
            examples=request.examples,
            metadata=request.metadata,
        )
        return datasets.register(dataset)

    @app.get("/metrics")
    def list_metrics() -> dict[str, list[str]]:
        """List available metric names.

        Returns:
            A document containing the registered metric names.
        """

        return {"metrics": metrics.list()}

    @app.post("/runs", response_model=Run, status_code=201)
    def create_run(request: CreateRunRequest) -> Run:
        """Execute an evaluation run.

        Args:
            request: Run execution payload.

        Returns:
            The completed run.

        Raises:
            HTTPException: If metrics or the dataset are invalid.
        """

        try:
            runner = Runner.from_names(request.metrics, registry=metrics)
        except KeyError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        if request.examples is not None:
            dataset = Dataset(
                name=request.dataset or "inline",
                version=request.dataset_version or "1",
                examples=request.examples,
            )
        elif request.dataset is not None:
            try:
                dataset = datasets.get(request.dataset, request.dataset_version)
            except KeyError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
        else:
            raise HTTPException(
                status_code=400,
                detail="either 'dataset' reference or inline 'examples' is required",
            )

        run = runner.run(dataset, name=request.name)
        if request.persist:
            run_store.save_run(run)
        return run

    @app.get("/runs", response_model=list[Run])
    def list_runs(limit: int = Query(default=50, ge=1, le=500)) -> list[Run]:
        """List stored runs, newest first.

        Args:
            limit: Maximum number of runs to return.

        Returns:
            A list of stored runs.
        """

        return run_store.list_runs(limit=limit)

    @app.get("/runs/{run_id}", response_model=Run)
    def get_run(run_id: str) -> Run:
        """Retrieve a stored run.

        Args:
            run_id: The run identifier.

        Returns:
            The stored run.

        Raises:
            HTTPException: If the run is not found.
        """

        try:
            return run_store.get_run(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/compare")
    def compare(request: CompareRequest) -> dict[str, Any]:
        """Compare two stored runs.

        Args:
            request: Baseline and candidate run identifiers.

        Returns:
            The comparison result.

        Raises:
            HTTPException: If either run is not found.
        """

        try:
            baseline = run_store.get_run(request.baseline_id)
            candidate = run_store.get_run(request.candidate_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        result = compare_runs(baseline, candidate)
        payload = result.model_dump()
        payload["counts"] = result.counts()
        return payload

    @app.post("/gates/evaluate", response_model=GateResult)
    def evaluate_gate_endpoint(request: GateRequest) -> GateResult:
        """Evaluate a stored run against a quality gate.

        Args:
            request: Gate evaluation payload.

        Returns:
            The gate result.

        Raises:
            HTTPException: If the run is not found.
        """

        try:
            run = run_store.get_run(request.run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        gate = QualityGate(
            metric_thresholds=request.metric_thresholds,
            overall_pass_rate=request.overall_pass_rate,
        )
        return gate.evaluate(run)

    return app


app = create_app()
