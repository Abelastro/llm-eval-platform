"""LLM/GenAI Evaluation & Regression Platform.

A deterministic, offline-first toolkit for scoring LLM and RAG outputs against
datasets, tracking runs, comparing versions, and gating CI on quality
thresholds.
"""

from llm_eval.config import Settings, get_settings
from llm_eval.models import (
    Dataset,
    Example,
    GateResult,
    MetricResult,
    Run,
)

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "Example",
    "GateResult",
    "MetricResult",
    "Run",
    "Settings",
    "get_settings",
    "__version__",
]
