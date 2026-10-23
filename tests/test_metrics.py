"""Tests for the metric implementations."""

from __future__ import annotations

import pytest

from llm_eval.metrics.base import clamp, lcs_length, ngrams, token_set, tokenize
from llm_eval.metrics.bleu import BLEUMetric
from llm_eval.metrics.contains import ContainsMetric
from llm_eval.metrics.exact_match import ExactMatchMetric
from llm_eval.metrics.faithfulness import FaithfulnessMetric, split_claims
from llm_eval.metrics.relevance import RelevanceMetric
from llm_eval.metrics.rouge import RougeLMetric
from llm_eval.models import Example


class TestTokenHelpers:
    """Tests for shared tokenization helpers."""

    def test_tokenize_lowercases_and_splits(self) -> None:
        assert tokenize("Hello, World! 42") == ["hello", "world", "42"]

    def test_tokenize_empty(self) -> None:
        assert tokenize("") == []

    def test_token_set_deduplicates(self) -> None:
        assert token_set("a a b") == {"a", "b"}

    def test_clamp_bounds(self) -> None:
        assert clamp(1.5) == 1.0
        assert clamp(-2.0) == 0.0
        assert clamp(0.3) == 0.3

    def test_ngrams_counts(self) -> None:
        assert ngrams(["a", "b", "a"], 2) == {("a", "b"): 1, ("b", "a"): 1}

    def test_ngrams_too_short(self) -> None:
        assert ngrams(["a"], 2) == {}

    def test_lcs_length(self) -> None:
        assert lcs_length(["a", "b", "c"], ["a", "c"]) == 2

    def test_lcs_empty(self) -> None:
        assert lcs_length([], ["a"]) == 0


class TestExactMatch:
    """Tests for :class:`ExactMatchMetric`."""

    def test_identical_scores_one(self) -> None:
        result = ExactMatchMetric().measure(Example(expected="Hello", actual="Hello"))
        assert result.score == 1.0
        assert result.passed is True

    def test_whitespace_normalized(self) -> None:
        result = ExactMatchMetric().measure(Example(expected="  Hello   world ", actual="hello world"))
        assert result.score == 1.0

    def test_case_sensitive_mismatch(self) -> None:
        metric = ExactMatchMetric(case_sensitive=True)
        result = metric.measure(Example(expected="Hello", actual="hello"))
        assert result.score == 0.0

    def test_missing_expected(self) -> None:
        result = ExactMatchMetric().measure(Example(actual="hello"))
        assert result.score == 0.0
        assert "no expected" in result.reason


class TestContains:
    """Tests for :class:`ContainsMetric`."""

    def test_substring_contained(self) -> None:
        result = ContainsMetric().measure(
            Example(expected="capital", actual="The capital of France")
        )
        assert result.score == 1.0

    def test_substring_absent(self) -> None:
        result = ContainsMetric().measure(Example(expected="capital", actual="a country"))
        assert result.score == 0.0

    def test_token_mode_partial_credit(self) -> None:
        result = ContainsMetric(mode="tokens").measure(
            Example(expected="the capital of France", actual="capital France")
        )
        assert result.score == pytest.approx(0.5)

    def test_token_mode_missing_expected(self) -> None:
        result = ContainsMetric(mode="tokens").measure(Example(actual="anything"))
        assert result.score == 0.0

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            ContainsMetric(mode="banana")


class TestBLEU:
    """Tests for :class:`BLEUMetric`."""

    def test_identical_scores_one(self) -> None:
        result = BLEUMetric().measure(
            Example(expected="the cat sat on the mat", actual="the cat sat on the mat")
        )
        assert result.score == pytest.approx(1.0)
        assert result.passed is True

    def test_empty_candidate(self) -> None:
        result = BLEUMetric().measure(Example(expected="hello world", actual=""))
        assert result.score == 0.0

    def test_no_expected(self) -> None:
        result = BLEUMetric().measure(Example(actual="hello"))
        assert result.score == 0.0

    def test_partial_overlap_between_zero_and_one(self) -> None:
        result = BLEUMetric().measure(
            Example(expected="the cat sat on the mat", actual="the cat sat on a rug")
        )
        assert 0.0 < result.score < 1.0

    def test_brevity_penalty_applied(self) -> None:
        long_ref = "the quick brown fox jumps over the lazy dog today"
        result = BLEUMetric().measure(Example(expected=long_ref, actual="the quick brown fox"))
        assert result.metadata["brevity_penalty"] < 1.0


class TestRouge:
    """Tests for :class:`RougeLMetric`."""

    def test_identical_scores_one(self) -> None:
        result = RougeLMetric().measure(
            Example(expected="the cat sat", actual="the cat sat")
        )
        assert result.score == pytest.approx(1.0)

    def test_no_overlap_scores_zero(self) -> None:
        result = RougeLMetric().measure(Example(expected="alpha beta", actual="gamma delta"))
        assert result.score == 0.0

    def test_partial_recall(self) -> None:
        result = RougeLMetric().measure(
            Example(expected="a b c d", actual="a b x y")
        )
        assert 0.0 < result.score < 1.0
        assert result.metadata["lcs"] == 2

    def test_empty_reference(self) -> None:
        result = RougeLMetric().measure(Example(expected="", actual="a b c"))
        assert result.score == 0.0


class TestFaithfulness:
    """Tests for :class:`FaithfulnessMetric`."""

    def test_split_claims(self) -> None:
        assert split_claims("One. Two! Three?") == ["One", "Two", "Three"]

    def test_split_claims_without_punctuation(self) -> None:
        assert split_claims("single claim") == ["single claim"]

    def test_grounded_answer_scores_high(self) -> None:
        example = Example(
            actual="Paris is the capital of France.",
            context=["France is a country. Its capital is Paris."],
        )
        result = FaithfulnessMetric().measure(example)
        assert result.score == 1.0

    def test_ungrounded_answer_scores_zero(self) -> None:
        example = Example(
            actual="Bananas grow underwater.",
            context=["France is a country. Its capital is Paris."],
        )
        result = FaithfulnessMetric().measure(example)
        assert result.score == 0.0

    def test_missing_context(self) -> None:
        result = FaithfulnessMetric().measure(Example(actual="anything"))
        assert result.score == 0.0


class TestRelevance:
    """Tests for :class:`RelevanceMetric`."""

    def test_full_overlap_scores_one(self) -> None:
        result = RelevanceMetric().measure(
            Example(actual="paris is the capital", context=["paris is the capital"])
        )
        assert result.score == pytest.approx(1.0)

    def test_no_overlap_scores_zero(self) -> None:
        result = RelevanceMetric().measure(
            Example(actual="bananas", context=["paris capital france"])
        )
        assert result.score == 0.0

    def test_partial_overlap(self) -> None:
        result = RelevanceMetric().measure(
            Example(actual="paris is a city", context=["paris is the capital of france"])
        )
        assert 0.0 < result.score < 1.0

    def test_missing_context(self) -> None:
        result = RelevanceMetric().measure(Example(actual="paris"))
        assert result.score == 0.0

    def test_include_input(self) -> None:
        metric = RelevanceMetric(include_input=True)
        result = metric.measure(
            Example(input="paris", actual="capital", context=["paris capital"])
        )
        assert result.score == pytest.approx(1.0)
