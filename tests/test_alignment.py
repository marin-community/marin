# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the marin.alignment module."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from marin.alignment.coverage import (
    compute_coverage_stats,
    generate_covering_configs,
    make_tags,
    slugify_axis_value,
)
from marin.alignment.generate_prompts import (
    PromptGenConfig,
    _extract_tag,
    _parse_concretize_response,
    _parse_extraction_response,
    _parse_variation_axes,
    load_spec,
    write_sharded_jsonl_gz,
)
from marin.alignment.generate_responses import (
    _build_messages,
)
from marin.alignment.inference_config import LiteLLMConfig, VLLMConfig
from marin.alignment.judge import _parse_judge_response
from marin.alignment.llm_client import LLMResponse, llm_chat, llm_chat_single
from marin.alignment.prompts.concretize import make_concretize_prompt
from marin.alignment.prompts.extract import make_extraction_prompt
from marin.alignment.prompts.judge import (
    build_compliance_judge_prompt,
    build_judge_system_prompt,
    format_examples_for_calibration,
)
from marin.alignment.prompts.understanding import (
    STANDARD_DEMOGRAPHIC_AXES,
    make_behavior_understanding_prompt,
    make_understanding_system_prompt,
)
from marin.alignment.types import (
    AuthorityLevel,
    ComplianceResult,
    Example,
    Statement,
    StatementType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_axes():
    return [
        {"axis": "complexity", "spectrum": ["simple", "moderate", "complex", "expert"]},
        {"axis": "tone", "spectrum": ["casual", "formal", "technical"]},
        {"axis": "culture", "spectrum": ["western", "east_asian", "south_asian"]},
    ]


@pytest.fixture()
def sample_statement():
    return Statement(
        id="be_helpful",
        text="The model should be helpful and provide accurate information.",
        type=StatementType.GUIDELINE,
        authority_level=AuthorityLevel.PLATFORM,
        section="Core Behavior",
        subsection="Helpfulness",
        examples=[
            Example(
                description="Simple factual query",
                user_query="What is the capital of France?",
                good_response="The capital of France is Paris.",
                bad_response="I don't know.",
            ),
        ],
    )


@pytest.fixture()
def sample_spec_jsonl(tmp_path):
    spec_path = tmp_path / "spec.jsonl"
    statements = [
        {
            "id": "be_helpful",
            "text": "The model should be helpful.",
            "type": "GUIDELINE",
            "authority_level": "PLATFORM",
            "section": "Core",
            "subsection": "Helpfulness",
            "metadata": {
                "examples": [
                    {
                        "description": "Basic help",
                        "user_query": "Help me",
                        "good_response": "Sure!",
                        "bad_response": "No.",
                    }
                ]
            },
        },
        {
            "id": "be_honest",
            "text": "The model should be honest.",
            "type": "REQUIREMENT",
            "authority_level": "PLATFORM",
            "section": "Core",
            "subsection": "Honesty",
            "metadata": {},
        },
    ]
    with open(spec_path, "w") as f:
        for stmt in statements:
            f.write(json.dumps(stmt) + "\n")
    return spec_path


@pytest.fixture()
def sample_prompt():
    return {
        "behavior_id": "be_helpful",
        "system_prompt": "You are a customer service agent.",
        "user_message": "I need help with my order.",
        "rubric": "GOOD: Helpful response. BAD: Dismissive.",
        "config_id": "cfg_000",
        "axis_config": {"complexity": "simple"},
        "tags": ["complexity:simple"],
    }


@pytest.fixture()
def mock_litellm_response():
    """Factory for creating mock litellm completion responses."""

    def _create(content: str, model: str = "openai/gpt-4.1"):
        mock_choice = MagicMock()
        mock_choice.message.content = content

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = model
        mock_response.usage = mock_usage
        return mock_response

    return _create


# ===========================================================================
# Tests: coverage.py
# ===========================================================================


class TestCoverageAlgorithm:
    def test_pairwise_covering_all_tuples_covered(self, sample_axes):
        configs = generate_covering_configs(sample_axes, t=2, seed=42)
        stats = compute_coverage_stats(configs, sample_axes, t=2)
        assert stats["covered_tuples"] == stats["total_tuples"]

    def test_three_way_covering_all_tuples_covered(self, sample_axes):
        configs = generate_covering_configs(sample_axes, t=3, seed=42)
        stats = compute_coverage_stats(configs, sample_axes, t=3)
        assert stats["covered_tuples"] == stats["total_tuples"]

    def test_empty_axes_returns_empty(self):
        configs = generate_covering_configs([], t=2, seed=42)
        assert configs == []

    def test_single_axis_returns_all_values(self):
        axes = [{"axis": "x", "spectrum": ["a", "b", "c"]}]
        configs = generate_covering_configs(axes, t=2, seed=42)
        values = {c["x"] for c in configs}
        assert values == {"a", "b", "c"}

    def test_fewer_axes_than_t_returns_cross_product(self):
        axes = [
            {"axis": "x", "spectrum": ["a", "b"]},
        ]
        configs = generate_covering_configs(axes, t=3, seed=42)
        assert len(configs) == 2  # just all values of the single axis

    def test_deterministic_with_same_seed(self, sample_axes):
        configs1 = generate_covering_configs(sample_axes, t=2, seed=123)
        configs2 = generate_covering_configs(sample_axes, t=2, seed=123)
        assert configs1 == configs2

    def test_different_seeds_may_differ(self, sample_axes):
        configs1 = generate_covering_configs(sample_axes, t=2, seed=1)
        configs2 = generate_covering_configs(sample_axes, t=2, seed=999)
        # They cover the same tuples but the actual configs may differ
        stats1 = compute_coverage_stats(configs1, sample_axes, t=2)
        stats2 = compute_coverage_stats(configs2, sample_axes, t=2)
        assert stats1["covered_tuples"] == stats2["covered_tuples"]

    def test_coverage_stats_per_axis_counts(self, sample_axes):
        configs = generate_covering_configs(sample_axes, t=2, seed=42)
        stats = compute_coverage_stats(configs, sample_axes, t=2)
        # Every axis value should appear at least once
        for axis in sample_axes:
            name = axis["axis"]
            for val in axis["spectrum"]:
                assert stats["per_axis_value_counts"][name][val] >= 1


class TestSlugifyAndTags:
    def test_slugify_standard_axis(self):
        assert slugify_axis_value("user_cultural_context", "East Asian") == "culture:east_asian"

    def test_slugify_custom_axis(self):
        assert slugify_axis_value("complexity", "Multi-Part Request") == "complexity:multi_part_request"

    def test_slugify_long_value_truncated(self):
        long_value = "a" * 100
        result = slugify_axis_value("x", long_value)
        # tag portion should be <= 40 chars
        assert len(result.split(":")[1]) <= 40

    def test_make_tags(self, sample_axes):
        config = {"complexity": "simple", "tone": "formal", "culture": "western"}
        tags = make_tags(config, sample_axes)
        assert len(tags) == 3
        assert "complexity:simple" in tags
        assert "tone:formal" in tags
        assert "culture:western" in tags


# ===========================================================================
# Tests: types.py
# ===========================================================================


class TestTypes:
    def test_statement_from_dict(self):
        data = {
            "id": "test_stmt",
            "text": "Be good.",
            "type": "GUIDELINE",
            "authority_level": "PLATFORM",
            "section": "Core",
            "subsection": "Basics",
            "metadata": {
                "examples": [
                    {
                        "description": "Test",
                        "user_query": "Hello",
                        "good_response": "Hi!",
                        "bad_response": "...",
                    }
                ]
            },
        }
        stmt = Statement.from_dict(data)
        assert stmt.id == "test_stmt"
        assert stmt.type == StatementType.GUIDELINE
        assert stmt.authority_level == AuthorityLevel.PLATFORM
        assert len(stmt.examples) == 1
        assert stmt.examples[0].user_query == "Hello"

    def test_statement_from_dict_no_examples(self):
        data = {
            "id": "bare",
            "text": "Minimal.",
            "type": "REQUIREMENT",
            "authority_level": "USER",
        }
        stmt = Statement.from_dict(data)
        assert stmt.examples == []
        assert stmt.section == ""

    def test_compliance_result_from_judge_output(self):
        output = {"score": 8, "confidence": 0.9, "explanation": "Good.", "highlights": ["quote"]}
        result = ComplianceResult.from_judge_output(output)
        assert result.score == 8
        assert result.compliant is True
        assert result.confidence == 0.9

    def test_compliance_result_non_compliant(self):
        output = {"score": 3, "confidence": 0.8, "explanation": "Bad."}
        result = ComplianceResult.from_judge_output(output)
        assert result.compliant is False


# ===========================================================================
# Tests: generate_prompts.py — parsing helpers
# ===========================================================================


class TestPromptParsing:
    def test_extract_tag(self):
        text = "prefix <behavior_understanding>This is the understanding.</behavior_understanding> suffix"
        assert _extract_tag(text, "behavior_understanding") == "This is the understanding."

    def test_extract_tag_missing(self):
        assert _extract_tag("no tags here", "missing") == ""

    def test_extract_tag_multiline(self):
        text = "<scientific_motivation>\nLine 1\nLine 2\n</scientific_motivation>"
        result = _extract_tag(text, "scientific_motivation")
        assert "Line 1" in result
        assert "Line 2" in result

    def test_parse_variation_axes_valid(self):
        text = """
        <variation_axes>
        [
            {"axis": "complexity", "spectrum": ["low", "medium", "high"], "description": "task complexity"},
            {"axis": "tone", "spectrum": ["formal", "informal"], "why_it_matters": "affects response style"}
        ]
        </variation_axes>
        """
        axes = _parse_variation_axes(text)
        assert len(axes) == 2
        assert axes[0]["axis"] == "complexity"
        assert len(axes[0]["spectrum"]) == 3

    def test_parse_variation_axes_missing_block(self):
        with pytest.raises(ValueError, match="missing"):
            _parse_variation_axes("no variation_axes here")

    def test_parse_variation_axes_invalid_json(self):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _parse_variation_axes("<variation_axes>not json</variation_axes>")

    def test_parse_variation_axes_too_few_spectrum_values(self):
        text = '<variation_axes>[{"axis": "x", "spectrum": ["only_one"]}]</variation_axes>'
        with pytest.raises(ValueError, match="spectrum"):
            _parse_variation_axes(text)

    def test_parse_concretize_response(self):
        content = """
        <scenario>A user asks about cooking pasta.</scenario>
        <rubric>GOOD: Detailed recipe. BAD: Vague answer.</rubric>
        <scenario>A user asks about astrophysics.</scenario>
        <rubric>GOOD: Clear explanation. BAD: Jargon-heavy.</rubric>
        """
        results = _parse_concretize_response(content)
        assert len(results) == 2
        assert "cooking pasta" in results[0]["description"]
        assert "GOOD" in results[1]["rubric"]

    def test_parse_extraction_response(self):
        content = """
        <scenario_0>
        <system_prompt>You are a helpful assistant.</system_prompt>
        <user_message>Tell me about Python.</user_message>
        </scenario_0>
        <scenario_1>
        <system_prompt>You are a chef.</system_prompt>
        <user_message>How do I make pasta?</user_message>
        </scenario_1>
        """
        results = _parse_extraction_response(content, batch_size=2, batch_start_idx=0)
        assert len(results) == 2
        assert results[0]["system_prompt"] == "You are a helpful assistant."
        assert results[0]["user_message"] == "Tell me about Python."
        assert results[1]["system_prompt"] == "You are a chef."

    def test_parse_extraction_response_missing_scenario(self):
        content = "<scenario_0><system_prompt>Hi</system_prompt><user_message>Hello</user_message></scenario_0>"
        with pytest.raises(RuntimeError, match="missing <scenario_1>"):
            _parse_extraction_response(content, batch_size=2, batch_start_idx=0)


# ===========================================================================
# Tests: generate_prompts.py — spec loading
# ===========================================================================


class TestSpecLoading:
    def test_load_spec_jsonl(self, sample_spec_jsonl):
        statements = load_spec(str(sample_spec_jsonl))
        assert len(statements) == 2
        assert "be_helpful" in statements
        assert "be_honest" in statements
        assert statements["be_helpful"].type == StatementType.GUIDELINE
        assert statements["be_honest"].type == StatementType.REQUIREMENT

    def test_load_spec_gzip(self, tmp_path):
        spec_path = tmp_path / "spec.jsonl.gz"
        stmt = {"id": "test", "text": "Test.", "type": "GUIDELINE", "authority_level": "PLATFORM", "metadata": {}}
        with gzip.open(spec_path, "wt", encoding="utf-8") as f:
            f.write(json.dumps(stmt) + "\n")
        statements = load_spec(str(spec_path))
        assert "test" in statements

    def test_load_spec_with_examples(self, sample_spec_jsonl):
        statements = load_spec(str(sample_spec_jsonl))
        helpful = statements["be_helpful"]
        assert len(helpful.examples) == 1
        assert helpful.examples[0].user_query == "Help me"


# ===========================================================================
# Tests: generate_prompts.py — sharded output
# ===========================================================================


class TestShardedOutput:
    def testwrite_sharded_jsonl_gz(self, tmp_path):
        records = [{"id": i, "text": f"record {i}"} for i in range(12)]
        write_sharded_jsonl_gz(records, str(tmp_path / "output"), shard_size=5)

        output_dir = tmp_path / "output"
        shards = sorted(output_dir.glob("*.jsonl.gz"))
        assert len(shards) == 3  # 5 + 5 + 2

        # Read back and verify
        all_records = []
        for shard in shards:
            with gzip.open(shard, "rt") as f:
                for line in f:
                    all_records.append(json.loads(line))
        assert len(all_records) == 12
        assert all_records[0]["id"] == 0
        assert all_records[11]["id"] == 11

    def test_write_sharded_empty(self, tmp_path):
        write_sharded_jsonl_gz([], str(tmp_path / "empty"), shard_size=5)
        shards = sorted((tmp_path / "empty").glob("*.jsonl.gz"))
        assert len(shards) == 1  # one empty shard


# ===========================================================================
# Tests: generate_responses.py — helpers
# ===========================================================================


class TestInferenceConfig:
    def test_litellm_config_is_api(self):
        config = LiteLLMConfig(model="openai/gpt-4.1")
        assert config.is_api is True
        assert config.is_local is False

    def test_vllm_config_is_local(self):
        config = VLLMConfig(model="/path/to/checkpoint")
        assert config.is_local is True
        assert config.is_api is False

    def test_litellm_config_resources_are_cpu(self):
        config = LiteLLMConfig(model="openai/gpt-4.1")
        resources = config.resources
        assert resources is not None

    def test_vllm_config_resources_are_tpu(self):
        config = VLLMConfig(model="/path/to/checkpoint", tpu_type="v6e-8")
        resources = config.resources
        assert resources is not None

    def test_litellm_config_defaults(self):
        config = LiteLLMConfig(model="openai/gpt-4.1")
        assert config.num_retries == 10
        assert config.workers == 64

    def test_vllm_config_defaults(self):
        config = VLLMConfig(model="/path/to/model")
        assert config.tensor_parallel_size == 1
        assert config.max_model_len == 4096
        assert config.gpu_memory_utilization == 0.9

    def test_vllm_config_custom(self):
        config = VLLMConfig(model="my-model", tensor_parallel_size=4, max_model_len=8192, tpu_type="v5p-32")
        assert config.tensor_parallel_size == 4
        assert config.max_model_len == 8192
        assert config.tpu_type == "v5p-32"


class TestResponseHelpers:
    def test_build_messages_with_spec_guidance(self, sample_prompt):
        behavior_statements = {"be_helpful": "Always be helpful and friendly."}
        messages = _build_messages(sample_prompt, behavior_statements)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "customer service agent" in messages[0]["content"]
        assert "Always be helpful" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_build_messages_without_spec_guidance(self, sample_prompt):
        messages = _build_messages(sample_prompt, None)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "behavioral guideline" not in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_build_messages_no_system_prompt(self):
        prompt = {"behavior_id": "x", "system_prompt": "", "user_message": "Hello"}
        messages = _build_messages(prompt, None)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


# ===========================================================================
# Tests: judge.py — parse judge response
# ===========================================================================


class TestJudgeParsing:
    def test_parse_json_in_code_block(self):
        content = '```json\n{"score": 8, "confidence": 0.9, "explanation": "Good", "highlights": []}\n```'
        result = _parse_judge_response(content)
        assert result["score"] == 8
        assert result["confidence"] == 0.9

    def test_parse_raw_json(self):
        content = '{"score": 3, "confidence": 0.5, "explanation": "Bad"}'
        result = _parse_judge_response(content)
        assert result["score"] == 3

    def test_parse_json_with_surrounding_text(self):
        content = 'Here is my judgment:\n{"score": 7, "confidence": 0.8, "explanation": "OK"}\nThat is all.'
        result = _parse_judge_response(content)
        assert result["score"] == 7

    def test_parse_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_judge_response("not json at all")


# ===========================================================================
# Tests: llm_client.py
# ===========================================================================


class TestLLMClient:
    @patch("marin.alignment.llm_client.litellm.completion")
    def test_llm_chat_returns_responses(self, mock_completion, mock_litellm_response):
        mock_completion.return_value = mock_litellm_response("Hello world!")
        responses = llm_chat(
            config=LiteLLMConfig(model="openai/gpt-4.1"),
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert len(responses) == 1
        assert responses[0].content == "Hello world!"
        assert responses[0].model == "openai/gpt-4.1"
        mock_completion.assert_called_once()

    @patch("marin.alignment.llm_client.litellm.completion")
    def test_llm_chat_with_string_config(self, mock_completion, mock_litellm_response):
        """Bare string is auto-converted to LiteLLMConfig."""
        mock_completion.return_value = mock_litellm_response("Response")
        responses = llm_chat(
            config="openai/gpt-4.1",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert len(responses) == 1
        assert responses[0].content == "Response"

    @patch("marin.alignment.llm_client.litellm.completion")
    def test_llm_chat_with_system_prompt(self, mock_completion, mock_litellm_response):
        mock_completion.return_value = mock_litellm_response("Response")
        llm_chat(
            config=LiteLLMConfig(model="openai/gpt-4.1"),
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="You are helpful.",
        )
        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"

    @patch("marin.alignment.llm_client.litellm.completion")
    def test_llm_chat_single(self, mock_completion, mock_litellm_response):
        mock_completion.return_value = mock_litellm_response("Single response")
        response = llm_chat_single(
            config=LiteLLMConfig(model="openai/gpt-4.1"),
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert isinstance(response, LLMResponse)
        assert response.content == "Single response"

    @patch("marin.alignment.llm_client.litellm.completion")
    def test_llm_chat_multiple_n(self, mock_completion):
        mock_choice1 = MagicMock()
        mock_choice1.message.content = "Response 1"
        mock_choice2 = MagicMock()
        mock_choice2.message.content = "Response 2"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice1, mock_choice2]
        mock_response.model = "openai/gpt-4.1"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_completion.return_value = mock_response

        responses = llm_chat(
            config=LiteLLMConfig(model="openai/gpt-4.1"),
            messages=[{"role": "user", "content": "Hi"}],
            n=2,
        )
        assert len(responses) == 2
        assert responses[0].content == "Response 1"
        assert responses[1].content == "Response 2"


# ===========================================================================
# Tests: prompt templates
# ===========================================================================


class TestPromptTemplates:
    def test_understanding_system_prompt(self):
        prompt = make_understanding_system_prompt()
        assert "alignment research assistant" in prompt
        assert "BloomUnderstanding" not in prompt

    def test_behavior_understanding_prompt(self):
        prompt = make_behavior_understanding_prompt("honesty", "The model should be honest.")
        assert "honesty" in prompt
        assert "variation_axes" in prompt
        assert "ORTHOGONAL" in prompt
        assert "demographic" in prompt.lower()

    def test_standard_demographic_axes_structure(self):
        assert len(STANDARD_DEMOGRAPHIC_AXES) == 2
        for axis in STANDARD_DEMOGRAPHIC_AXES:
            assert "axis" in axis
            assert "spectrum" in axis
            assert len(axis["spectrum"]) >= 4
            assert axis.get("standard") is True

    def test_concretize_prompt(self):
        system_prompt, user_prompt = make_concretize_prompt(
            behavior_name="helpfulness",
            behavior_understanding="The model helps users.",
            scientific_motivation="Testing assistance quality.",
            transcript_analyses=[],
            configs=[{"complexity": "simple", "tone": "casual"}],
            axes_metadata=[
                {"axis": "complexity", "spectrum": ["simple", "complex"], "description": "Task complexity"},
                {"axis": "tone", "spectrum": ["casual", "formal"], "description": "Response tone"},
            ],
        )
        assert "evaluation scenario designer" in system_prompt
        assert "helpfulness" in user_prompt
        assert "Configuration 1" in user_prompt
        assert "simple" in user_prompt

    def test_extraction_prompt_with_system(self):
        system_prompt, user_prompt = make_extraction_prompt(
            scenarios=[{"description": "A user asks about weather."}],
            batch_start_idx=0,
            include_system_prompt=True,
        )
        assert "system_prompt" in system_prompt
        assert "user_message" in system_prompt
        assert "Scenario 0" in user_prompt

    def test_extraction_prompt_without_system(self):
        system_prompt, _ = make_extraction_prompt(
            scenarios=[{"description": "A user asks about weather."}],
            batch_start_idx=0,
            include_system_prompt=False,
        )
        assert "system_prompt" not in system_prompt
        assert "user_message" in system_prompt

    def test_judge_system_prompt(self):
        prompt = build_judge_system_prompt()
        assert "expert evaluator" in prompt
        assert "1-10" in prompt

    def test_compliance_judge_prompt(self, sample_statement):
        prompt = build_compliance_judge_prompt(
            statement=sample_statement,
            user_input="What is the capital of France?",
            model_response="Paris is the capital of France.",
        )
        assert "be_helpful" in prompt
        assert "GUIDELINE" in prompt
        assert "capital of France" in prompt
        assert "Scoring Guide" in prompt

    def test_compliance_judge_prompt_with_rubric(self, sample_statement):
        prompt = build_compliance_judge_prompt(
            statement=sample_statement,
            user_input="Hello",
            model_response="Hi there!",
            question_rubric="GOOD: Warm greeting. BAD: Cold response.",
        )
        assert "Question-Specific Criteria" in prompt
        assert "Warm greeting" in prompt

    def test_calibration_examples_formatted(self, sample_statement):
        text = format_examples_for_calibration(sample_statement)
        assert "Calibration Examples" in text
        assert "capital of France" in text
        assert "Compliant Response" in text

    def test_calibration_examples_empty(self):
        stmt = Statement(
            id="no_examples",
            text="Test.",
            type=StatementType.GUIDELINE,
            authority_level=AuthorityLevel.PLATFORM,
            section="",
            subsection="",
        )
        assert format_examples_for_calibration(stmt) == ""


# ===========================================================================
# Tests: end-to-end prompt generation (mocked LLM)
# ===========================================================================


class TestPromptGenerationE2E:
    """Test the full 3-stage pipeline with mocked LLM calls."""

    @patch("marin.alignment.generate_prompts.llm_chat_single")
    def test_full_pipeline(self, mock_llm, sample_spec_jsonl, tmp_path):
        """Test generate_prompts_from_spec with all 3 stages mocked."""
        # Stage 1: Understanding response
        stage1_response = LLMResponse(
            content=(
                "<behavior_understanding>Being helpful means assisting users.</behavior_understanding>\n"
                "<scientific_motivation>Testing helpfulness is important.</scientific_motivation>\n"
                "<variation_axes>\n"
                "[\n"
                '  {"axis": "request_complexity", "spectrum": ["simple", "moderate", "complex"], '
                '"description": "How complex the request is"},\n'
                '  {"axis": "domain", "spectrum": ["general", "technical", "creative"], '
                '"description": "The domain of the request"}\n'
                "]\n"
                "</variation_axes>"
            ),
            model="openai/gpt-4.1",
        )

        # Stage 2: Concretize response
        stage2_response = LLMResponse(
            content=(
                "<scenario>A student asks for help with basic math homework. "
                '"You are a math tutor." The student says "Can you help me with fractions?"</scenario>\n'
                "<rubric>GOOD: Patient explanation. BAD: Dismissive.</rubric>\n"
            ),
            model="openai/gpt-4.1",
        )

        # Dispatch mock based on system_prompt content:
        # - Stage 1: understanding system prompt contains "alignment research assistant"
        # - Stage 2: concretize system prompt contains "scenario designer"
        # - Stage 3: extraction — returns <scenario_N> with correct index

        def mock_side_effect(**kwargs):
            sys_prompt = kwargs.get("system_prompt") or ""
            messages = kwargs.get("messages") or []

            if "alignment research assistant" in sys_prompt:
                return stage1_response
            elif "scenario designer" in sys_prompt:
                return stage2_response
            else:
                # Extraction: parse the batch start index from the user prompt
                # The extraction prompt contains "Scenario N:" lines
                user_content = messages[0]["content"] if messages else ""
                idx_match = re.search(r"Scenario (\d+)", user_content)
                idx = int(idx_match.group(1)) if idx_match else 0
                return LLMResponse(
                    content=(
                        f"<scenario_{idx}>\n"
                        "<system_prompt>You are a math tutor.</system_prompt>\n"
                        "<user_message>Can you help me with fractions?</user_message>\n"
                        f"</scenario_{idx}>\n"
                    ),
                    model="openai/gpt-4.1-mini",
                )

        mock_llm.side_effect = mock_side_effect

        output_path = str(tmp_path / "prompts_output")

        config = PromptGenConfig(
            spec_path=str(sample_spec_jsonl),
            output_path=output_path,
            ideation_model="openai/gpt-4.1",
            extract_model="openai/gpt-4.1-mini",
            covering_strength=2,
            concretize_batch_size=1,
            extract_batch_size=1,
            ideation_workers=1,
            concretize_workers=1,
            extract_workers=1,
        )

        from marin.alignment.generate_prompts import generate_prompts_from_spec

        generate_prompts_from_spec(config)

        # Verify output was written
        output_dir = Path(output_path)
        assert output_dir.exists()
        shards = list(output_dir.glob("*.jsonl.gz"))
        assert len(shards) >= 1

        # Read back prompts
        all_prompts = []
        for shard in shards:
            with gzip.open(shard, "rt") as f:
                for line in f:
                    if line.strip():
                        all_prompts.append(json.loads(line))

        # Should have generated prompts for both statements
        assert len(all_prompts) > 0
        # Each prompt should have the expected fields
        for prompt in all_prompts:
            assert "behavior_id" in prompt
            assert "system_prompt" in prompt
            assert "user_message" in prompt
            assert "rubric" in prompt


# ===========================================================================
# Tests: judge.py — end-to-end pair construction (mocked)
# ===========================================================================


class TestJudgeE2E:
    @patch("marin.alignment.judge.llm_chat_single")
    def test_process_prompt_pair_accepted(self, mock_llm, sample_spec_jsonl):
        """Test that a good chosen + bad rejected pair passes the filter."""
        from marin.alignment.judge import _process_prompt_pair

        statements = load_spec(str(sample_spec_jsonl))

        # High score for chosen, low for rejected
        call_count = {"n": 0}

        def mock_side_effect(**kwargs):
            call_count["n"] += 1
            # First call is for chosen, rest are for rejected
            if call_count["n"] == 1:
                score = 9
            else:
                score = 2
            return LLMResponse(
                content=json.dumps({"score": score, "confidence": 0.9, "explanation": "Test", "highlights": []}),
                model="openai/gpt-4.1",
            )

        mock_llm.side_effect = mock_side_effect

        chosen_record = {
            "prompt_id": "be_helpful/cfg_000",
            "behavior_id": "be_helpful",
            "system_prompt": "You are helpful.",
            "user_message": "Help me.",
            "rubric": "GOOD: Helps. BAD: Refuses.",
            "responses": [{"content": "Of course, I'd be happy to help!", "index": 0}],
        }
        rejected_record = {
            "prompt_id": "be_helpful/cfg_000",
            "behavior_id": "be_helpful",
            "system_prompt": "You are helpful.",
            "user_message": "Help me.",
            "rubric": "GOOD: Helps. BAD: Refuses.",
            "responses": [{"content": "No, figure it out yourself.", "index": 0}],
        }

        from marin.alignment.judge import JudgePairConfig

        config = JudgePairConfig(
            prompts_path="",
            chosen_responses_path="",
            rejected_responses_path="",
            spec_path="",
            output_path="",
            min_chosen_score=7.0,
            min_gap=2.0,
        )

        pair = _process_prompt_pair("be_helpful/cfg_000", chosen_record, rejected_record, statements, config)

        assert pair is not None
        assert len(pair["chosen"]) == 3  # system + user + assistant
        assert len(pair["rejected"]) == 3
        assert pair["chosen"][0]["role"] == "system"
        assert pair["chosen"][2]["content"] == "Of course, I'd be happy to help!"
        assert pair["rejected"][2]["content"] == "No, figure it out yourself."

    @patch("marin.alignment.judge.llm_chat_single")
    def test_process_prompt_pair_filtered_low_chosen(self, mock_llm, sample_spec_jsonl):
        """Test that a low-scoring chosen response gets filtered out."""
        from marin.alignment.judge import JudgePairConfig, _process_prompt_pair

        statements = load_spec(str(sample_spec_jsonl))

        mock_llm.return_value = LLMResponse(
            content=json.dumps({"score": 4, "confidence": 0.8, "explanation": "Poor", "highlights": []}),
            model="openai/gpt-4.1",
        )

        chosen_record = {
            "behavior_id": "be_helpful",
            "system_prompt": "",
            "user_message": "Help",
            "rubric": "",
            "responses": [{"content": "Meh.", "index": 0}],
        }
        rejected_record = {
            "behavior_id": "be_helpful",
            "responses": [{"content": "No.", "index": 0}],
        }

        config = JudgePairConfig(
            prompts_path="",
            chosen_responses_path="",
            rejected_responses_path="",
            spec_path="",
            output_path="",
            min_chosen_score=7.0,
            min_gap=2.0,
        )

        pair = _process_prompt_pair("test", chosen_record, rejected_record, statements, config)
        assert pair is None  # filtered out due to low chosen score

    @patch("marin.alignment.judge.llm_chat_single")
    def test_process_prompt_pair_filtered_small_gap(self, mock_llm, sample_spec_jsonl):
        """Test that similar chosen/rejected scores get filtered out."""
        from marin.alignment.judge import JudgePairConfig, _process_prompt_pair

        statements = load_spec(str(sample_spec_jsonl))

        # Both score similarly
        mock_llm.return_value = LLMResponse(
            content=json.dumps({"score": 7, "confidence": 0.8, "explanation": "OK", "highlights": []}),
            model="openai/gpt-4.1",
        )

        chosen_record = {
            "behavior_id": "be_helpful",
            "system_prompt": "",
            "user_message": "Help",
            "rubric": "",
            "responses": [{"content": "Sure.", "index": 0}],
        }
        rejected_record = {
            "behavior_id": "be_helpful",
            "responses": [{"content": "OK.", "index": 0}],
        }

        config = JudgePairConfig(
            prompts_path="",
            chosen_responses_path="",
            rejected_responses_path="",
            spec_path="",
            output_path="",
            min_chosen_score=7.0,
            min_gap=2.0,
        )

        pair = _process_prompt_pair("test", chosen_record, rejected_record, statements, config)
        assert pair is None  # filtered: gap = 0 < 2.0
