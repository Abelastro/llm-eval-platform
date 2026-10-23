"""Application settings loaded from environment variables.

All runtime configuration is namespaced under the ``LLM_EVAL_`` prefix so the
platform can be embedded without clashing with unrelated environment state.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the evaluation platform.

    Attributes:
        app_name: Human readable service name used in logs and API metadata.
        store_path: Filesystem location of the SQLite run store.
        default_pass_threshold: Score at or above which a metric result passes.
        default_overall_pass_rate: Minimum fraction of passing examples for an
            overall gate pass when no explicit floor is provided.
        judge_model: Model identifier handed to the optional LLM judge client.
        openai_api_key: [REDACTED] API key; when unset the deterministic
            heuristic judge is used instead of any network client.
        default_dataset_dir: Directory scanned for bundled JSONL/CSV datasets.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "llm-eval-platform"
    store_path: str = "llm_eval_runs.sqlite3"
    default_pass_threshold: float = 0.5
    default_overall_pass_rate: float = 0.6
    judge_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    default_dataset_dir: str = "examples"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Returns:
        The process-wide settings object, constructed on first access.
    """

    return Settings()
