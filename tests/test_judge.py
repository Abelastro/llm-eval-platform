"""Tests for the LLM judge and rubric."""

from __future__ import annotations

import pytest

from llm_eval.judges.llm_judge import (
    HeuristicJudgeClient,
    JudgeClient,
    LLMJudge,
    OpenAIChatJudgeClient,
)
from llm_eval.judges.rubric import Rubric, RubricCriterion, default_rubric
from llm_eval.models import Example


class TestRubric:
    """Tests for rubric models."""

    def test_default_rubric_has_criteria(self) -> None:
        rubric = default_rubric()
        assert len(rubric.criteria) == 3
        assert "correctness" in {c.name for c in rubric.criteria}

    def test_normalized_weights_sum_to_one(self) -> None:
        weights = default_rubric().normalized_weights()
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_normalized_weights_empty(self) -> None:
        assert Rubric().normalized_weights() == {}

    def test_prompt_renders_criteria(self) -> None:
        prompt = default_rubric().prompt()
        assert "correctness" in prompt
        assert "Rubric:" in prompt


class TestHeuristicJudgeClient:
    """Tests for the deterministic judge client."""

    def test_protocol_conformance(self) -> None:
        assert isinstance(HeuristicJudgeClient(), JudgeClient)

    def test_correctness_full_overlap(self) -> None:
        client = HeuristicJudgeClient()
        criterion = RubricCriterion(name="correctness")
        score = client.score(
            criterion, Example(expected="paris capital", actual="paris capital")
        )
        assert score == pytest.approx(1.0)

    def test_correctness_no_overlap(self) -> None:
        client = HeuristicJudgeClient()
        score = client.score(
            RubricCriterion(name="correctness"), Example(expected="paris", actual="banana")
        )
        assert score == 0.0

    def test_clarity_rewards_complete_sentence(self) -> None:
        client = HeuristicJudgeClient()
        score = client.score(
            RubricCriterion(name="clarity"),
            Example(actual="This is a well formed answer about the topic."),
        )
        assert score > 0.5

    def test_unknown_criterion_falls_back(self) -> None:
        client = HeuristicJudgeClient()
        score = client.score(
            RubricCriterion(name="vibes"), Example(expected="a", actual="a")
        )
        assert score == pytest.approx(1.0)


class TestLLMJudge:
    """Tests for the metric-style judge."""

    def test_weighted_score_in_range(self) -> None:
        result = LLMJudge().measure(
            Example(expected="paris is the capital", actual="paris is the capital")
        )
        assert 0.0 <= result.score <= 1.0

    def test_identical_answer_passes(self) -> None:
        result = LLMJudge().measure(
            Example(expected="the answer is paris", actual="the answer is paris.")
        )
        assert result.passed is True

    def test_empty_rubric_returns_zero(self) -> None:
        judge = LLMJudge(rubric=Rubric(name="empty"))
        result = judge.measure(Example(actual="x"))
        assert result.score == 0.0

    def test_custom_client_used(self) -> None:
        class FixedClient:
            def score(self, criterion: RubricCriterion, example: Example) -> float:
                return 0.5

        judge = LLMJudge(client=FixedClient())
        result = judge.measure(Example(actual="x"))
        assert result.score == pytest.approx(0.5)


class TestOpenAIJudgeStub:
    """Tests for the offline OpenAI judge stub."""

    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError):
            OpenAIChatJudgeClient(api_key="")

    def test_score_raises_without_transport(self) -> None:
        client = OpenAIChatJudgeClient(api_key="[REDACTED]")
        with pytest.raises(RuntimeError):
            client.score(RubricCriterion(name="correctness"), Example(actual="x"))
