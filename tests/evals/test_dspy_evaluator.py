"""Tests for DspyEvaluator Prime Intellect environment integration."""

from unittest.mock import MagicMock, patch

import dspy
import pytest
from dspy.utils.exceptions import AdapterParseError

from marin.evaluation.evaluators.dspy_evaluator import (
    _detect_format_error,
    _is_prime_intellect_task,
    _normalize_numeric,
    _parse_prime_intellect_env_id,
    _prime_intellect_metric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestPrimeIntellectTaskParsing:
    def test_is_prime_intellect_task(self):
        assert _is_prime_intellect_task("prime_intellect:gsm8k")
        assert _is_prime_intellect_task("prime_intellect:math500")
        assert not _is_prime_intellect_task("hover")
        assert not _is_prime_intellect_task("hotpotqa")

    def test_parse_env_id(self):
        assert _parse_prime_intellect_env_id("prime_intellect:gsm8k") == "gsm8k"
        assert _parse_prime_intellect_env_id("prime_intellect:math500") == "math500"
        assert _parse_prime_intellect_env_id("prime_intellect:gpqa_diamond") == "gpqa_diamond"


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class TestNormalizeNumeric:
    def test_integer(self):
        assert _normalize_numeric("400") == "400"

    def test_float_that_is_integer(self):
        assert _normalize_numeric("400.0") == "400"

    def test_float(self):
        assert _normalize_numeric("3.14") == "3.14"

    def test_with_commas(self):
        assert _normalize_numeric("1,000") == "1000"

    def test_non_numeric(self):
        assert _normalize_numeric("hello") is None

    def test_trailing_dot(self):
        assert _normalize_numeric("400.") == "400"


class TestPrimeIntellectMetric:
    def test_exact_match(self):
        example = dspy.Example(question="What is 2+2?", answer="4").with_inputs("question")
        pred = dspy.Prediction(answer="4")
        assert _prime_intellect_metric(example, pred) == 1.0

    def test_numeric_normalization(self):
        example = dspy.Example(question="Q", answer="400").with_inputs("question")
        pred = dspy.Prediction(answer="400.0")
        assert _prime_intellect_metric(example, pred) == 1.0

    def test_boxed_extraction(self):
        example = dspy.Example(question="Q", answer="42").with_inputs("question")
        pred = dspy.Prediction(answer="The answer is \\boxed{42}")
        assert _prime_intellect_metric(example, pred) == 1.0

    def test_hash_extraction(self):
        example = dspy.Example(question="Q", answer="25").with_inputs("question")
        pred = dspy.Prediction(answer="#### 25")
        assert _prime_intellect_metric(example, pred) == 1.0

    def test_wrong_answer(self):
        example = dspy.Example(question="Q", answer="42").with_inputs("question")
        pred = dspy.Prediction(answer="99")
        assert _prime_intellect_metric(example, pred) == 0.0

    def test_none_prediction(self):
        example = dspy.Example(question="Q", answer="42").with_inputs("question")
        assert _prime_intellect_metric(example, None) == 0.0

    def test_case_insensitive(self):
        example = dspy.Example(question="Q", answer="Paris").with_inputs("question")
        pred = dspy.Prediction(answer="paris")
        assert _prime_intellect_metric(example, pred) == 1.0


# ---------------------------------------------------------------------------
# Format-error detection
# ---------------------------------------------------------------------------


class TestDetectFormatError:
    """Unified format-error detection.

    _detect_format_error covers the *silent* failure case (ToonAdapter sets
    fields to None).  AdapterParseError from ChatAdapter/BAMLAdapter is
    caught explicitly in the eval loop -- see TestAdapterParseErrorHandling.
    """

    def test_pi_task_missing_answer(self):
        pred = dspy.Prediction(answer=None)
        assert _detect_format_error(pred, "prime_intellect:gsm8k") is True

    def test_pi_task_empty_answer(self):
        pred = dspy.Prediction(answer="")
        assert _detect_format_error(pred, "prime_intellect:gsm8k") is True

    def test_pi_task_valid_answer(self):
        pred = dspy.Prediction(answer="42")
        assert _detect_format_error(pred, "prime_intellect:gsm8k") is False

    def test_pi_task_none_prediction(self):
        assert _detect_format_error(None, "prime_intellect:gsm8k") is True

    def test_hover_missing_label(self):
        pred = dspy.Prediction(label=None)
        assert _detect_format_error(pred, "hover") is True

    def test_hover_valid_label(self):
        pred = dspy.Prediction(label="SUPPORTED")
        assert _detect_format_error(pred, "hover") is False

    def test_hotpotqa_missing_answer(self):
        pred = dspy.Prediction(answer=None)
        assert _detect_format_error(pred, "hotpotqa") is True

    def test_hotpotqa_valid_answer(self):
        pred = dspy.Prediction(answer="Paris")
        assert _detect_format_error(pred, "hotpotqa") is False

    def test_unknown_task_no_required_fields(self):
        pred = dspy.Prediction(whatever="value")
        assert _detect_format_error(pred, "unknown_task") is False


class TestAdapterParseError:
    """Verify that AdapterParseError carries the info we rely on."""

    def test_adapter_parse_error_is_catchable(self):
        sig = dspy.Signature("question -> answer")
        err = AdapterParseError(
            adapter_name="ChatAdapter",
            signature=sig,
            lm_response="garbage output",
        )
        assert isinstance(err, Exception)
        assert "ChatAdapter" in str(err)
        assert err.parsed_result is None

    def test_adapter_parse_error_with_partial_result(self):
        sig = dspy.Signature("question -> answer, reasoning")
        err = AdapterParseError(
            adapter_name="BAMLAdapter",
            signature=sig,
            lm_response='{"reasoning": "some text"}',
            parsed_result={"reasoning": "some text"},
        )
        assert err.parsed_result == {"reasoning": "some text"}
        assert "answer" in str(err)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


class TestLoadPrimeIntellect:
    def test_load_builtin_dataset(self):
        """Load a small slice of the real gsm8k dataset via verifiers."""
        from marin.evaluation.evaluators.dspy_evaluator import _load_prime_intellect

        examples = _load_prime_intellect("gsm8k", "test", 3)

        assert len(examples) == 3
        for ex in examples:
            assert hasattr(ex, "question")
            assert hasattr(ex, "answer")
            assert len(ex.question) > 0
            assert len(ex.answer) > 0

    @patch("verifiers.load_environment")
    @patch("verifiers.load_example_dataset", side_effect=ValueError("Not built-in"))
    def test_fallback_to_load_environment(self, mock_load_ds, mock_load_env):
        from marin.evaluation.evaluators.dspy_evaluator import _load_prime_intellect

        mock_env = MagicMock()
        mock_env.eval_dataset = True
        mock_env.get_eval_dataset.return_value = [
            {"question": "Custom Q", "answer": "Custom A"},
        ]
        mock_load_env.return_value = mock_env

        examples = _load_prime_intellect("custom_env", "test", 1)

        assert len(examples) == 1
        assert examples[0].question == "Custom Q"


# ---------------------------------------------------------------------------
# DspyEvaluator init
# ---------------------------------------------------------------------------


class TestDspyEvaluatorInit:
    def test_accepts_pi_task_name(self):
        from marin.evaluation.evaluators.dspy_evaluator import DspyEvaluator
        from marin.evaluation.evaluators.evaluator import ModelConfig

        evaluator = DspyEvaluator(
            model=ModelConfig(name="test-model", path=None, engine_kwargs={}),
            output_path="/tmp/test",
            adapter_name="chat",
            task_name="prime_intellect:gsm8k",
            endpoint="http://example.com:8000",
        )
        assert evaluator.task_name == "prime_intellect:gsm8k"

    def test_rejects_unknown_task(self):
        from marin.evaluation.evaluators.dspy_evaluator import DspyEvaluator
        from marin.evaluation.evaluators.evaluator import ModelConfig

        with pytest.raises(ValueError, match="Unknown task"):
            DspyEvaluator(
                model=ModelConfig(name="test-model", path=None, engine_kwargs={}),
                output_path="/tmp/test",
                adapter_name="chat",
                task_name="nonexistent_task",
                endpoint="http://example.com:8000",
            )

    def test_num_hops_stored(self):
        from marin.evaluation.evaluators.dspy_evaluator import DspyEvaluator
        from marin.evaluation.evaluators.evaluator import ModelConfig

        evaluator = DspyEvaluator(
            model=ModelConfig(name="test-model", path=None, engine_kwargs={}),
            output_path="/tmp/test",
            adapter_name="chat",
            task_name="prime_intellect:gsm8k",
            endpoint="http://example.com:8000",
            num_hops=2,
        )
        assert evaluator.num_hops == 2


# ---------------------------------------------------------------------------
# PrimeIntellectSolver program
# ---------------------------------------------------------------------------


class TestPrimeIntellectSolver:
    def test_single_turn_init(self):
        from experiments.dspy.programs.prime_intellect import PrimeIntellectSolver

        solver = PrimeIntellectSolver()
        assert solver.num_hops == 0
        assert solver.search is None

    def test_multi_hop_init(self):
        from experiments.dspy.programs.prime_intellect import PrimeIntellectSolver

        def mock_search(q, k):
            return {}

        solver = PrimeIntellectSolver(search=mock_search, num_hops=2)
        assert solver.num_hops == 2
        assert solver.search is mock_search
