# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the marin.alignment module."""

from __future__ import annotations

import gzip
import json
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import marin.alignment.llm_client as llm_client_module
from fray.v2.types import ResourceConfig
from marin.alignment.align import AlignConfig, ResponseExecutionMode, align
from marin.alignment.batched_vllm_serve import BatchedVllmServeSession, _build_model_config
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
    _run_concretization_stage_local,
    load_sharded_jsonl_gz,
    load_spec,
    write_sharded_jsonl_gz,
)
from marin.alignment.generate_responses import (
    RejectedPromptStrategy,
    ResponseRole,
    _build_chosen_messages,
    _build_rejected_messages,
    ResponsePairGenConfig,
    ResponseGenConfig,
    generate_response_pair,
    generate_responses,
)
from marin.alignment.inference_config import OpenAIConfig, VLLMConfig
from marin.alignment.judge import (
    PreferencePairFilterConfig,
    JudgeConfig,
    parse_compliance_result,
    parse_judge_response,
    build_preference_pairs,
    judge_responses,
)
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
from marin.inference.vllm_server import _engine_kwargs_to_cli_args
from marin.alignment.types import (
    AuthorityLevel,
    ComplianceResult,
    Example,
    Statement,
    StatementType,
)
from marin.execution.executor import ExecutorStep, InputName, output_path_of
from marin.execution.remote import remote

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _noop_remote(_config) -> None:
    return None


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
def mock_openai_response():
    """Factory for creating mock OpenAI chat completion responses."""

    def _create(content: str, model: str = "gpt-4.1"):
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
        """
        result = _parse_concretize_response(content)
        assert "cooking pasta" in result["description"]
        assert "GOOD" in result["rubric"]

    def test_parse_extraction_response(self):
        content = """
        <system_prompt>You are a helpful assistant.</system_prompt>
        <user_message>Tell me about Python.</user_message>
        """
        result = _parse_extraction_response(content)
        assert result["system_prompt"] == "You are a helpful assistant."
        assert result["user_message"] == "Tell me about Python."

    def test_parse_extraction_response_missing_user_message(self):
        content = "<system_prompt>Hi</system_prompt>"
        with pytest.raises(RuntimeError, match="missing <user_message>"):
            _parse_extraction_response(content)


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
    class _FakeRequestsResponse:
        def __init__(self, payload: dict[str, object], *, status_code: int = 200):
            self._payload = payload
            self.status_code = status_code
            self.text = json.dumps(payload)

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self) -> dict[str, object]:
            return self._payload

    def test_openai_config_is_api(self):
        config = OpenAIConfig(model="gpt-4.1")
        assert config.is_api is True
        assert config.is_local is False

    def test_vllm_config_is_local(self):
        config = VLLMConfig(model="/path/to/checkpoint")
        assert config.is_local is True
        assert config.is_api is False

    def test_openai_config_resources_are_cpu(self):
        config = OpenAIConfig(model="gpt-4.1")
        resources = config.resources
        assert resources is not None

    def test_vllm_config_resources_are_tpu(self):
        config = VLLMConfig(model="/path/to/checkpoint", tpu_type="v6e-8")
        resources = config.resources
        assert resources is not None

    def test_vllm_config_resources_are_gpu_when_gpu_type_set(self):
        config = VLLMConfig(model="/path/to/checkpoint", gpu_type="H100", gpu_count=2, tpu_type=None)
        resources = config.resources
        assert resources.device.variant == "H100"
        assert resources.device.count == 2

    def test_openai_config_defaults(self):
        config = OpenAIConfig(model="gpt-4.1")
        assert config.num_retries == 10
        assert config.workers == 64

    def test_vllm_config_defaults(self):
        config = VLLMConfig(model="/path/to/model")
        assert config.tokenizer is None
        assert config.hf_overrides is None
        assert config.tensor_parallel_size == 1
        assert config.max_model_len == 4096
        assert config.gpu_memory_utilization == 0.9
        assert config.load_format is None
        assert config.gpu_type is None
        assert config.native_stderr_mode == "file"
        assert config.resolved_serve_mode == "native"
        assert config.pip_dependency_groups == ["vllm", "tpu"]

    def test_vllm_config_custom(self):
        config = VLLMConfig(
            model="my-model",
            tokenizer="tokenizer-path",
            hf_overrides={"model_type": "gpt_oss"},
            tensor_parallel_size=4,
            max_model_len=8192,
            tpu_type="v5p-32",
            native_stderr_mode="tee",
        )
        assert config.tokenizer == "tokenizer-path"
        assert config.hf_overrides == {"model_type": "gpt_oss"}
        assert config.tensor_parallel_size == 4
        assert config.max_model_len == 8192
        assert config.tpu_type == "v5p-32"
        assert config.native_stderr_mode == "tee"

    def test_vllm_config_gpu_defaults_to_docker_mode(self):
        config = VLLMConfig(model="gs://bucket/model", gpu_type="H100", gpu_count=1, tpu_type=None)
        assert config.resolved_serve_mode == "docker"
        assert config.pip_dependency_groups == ["gpu"]

    def test_build_model_config_includes_tokenizer_override(self):
        config = VLLMConfig(
            model="gs://bucket/model/original",
            tokenizer="gs://bucket/model",
            hf_overrides={"model_type": "gpt_oss"},
            tensor_parallel_size=4,
            max_model_len=8192,
        )

        model_config = _build_model_config(config)

        assert model_config.path == "gs://bucket/model/original"
        assert model_config.engine_kwargs["tokenizer"] == "gs://bucket/model"
        assert model_config.engine_kwargs["hf_overrides"] == {"model_type": "gpt_oss"}

    def test_engine_kwargs_to_cli_args_include_tokenizer(self):
        args = _engine_kwargs_to_cli_args(
            {
                "tokenizer": "gs://bucket/model",
                "hf_overrides": {"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss"},
                "load_format": "runai_streamer",
                "tensor_parallel_size": 4,
            }
        )

        assert args[:2] == ["--tokenizer", "gs://bucket/model"]
        assert "--hf-overrides" in args
        assert "--load-format" in args

    def test_batched_vllm_rejects_completion_budget_without_prompt_room(self, monkeypatch):
        class FakeTokenizer:
            def __call__(self, texts, add_special_tokens=False):
                assert add_special_tokens is False
                return {"input_ids": [[0] * 944 for _ in texts]}

        session = BatchedVllmServeSession(VLLMConfig(model="gs://bucket/model", max_model_len=4096))
        session._env = types.SimpleNamespace(model_id="fake-model", server_url="http://127.0.0.1:8000/v1")
        session._tokenizer = FakeTokenizer()

        def fail_post(*args, **kwargs):
            raise AssertionError("generate_from_prompt_texts should fail before making an HTTP request")

        monkeypatch.setattr("marin.alignment.batched_vllm_serve.requests.post", fail_post)

        with pytest.raises(ValueError, match="leaves no room for prompt tokens"):
            session.generate_from_prompt_texts(
                ["prompt"],
                stage_name="understanding",
                temperature=0.0,
                max_tokens=4096,
                n=1,
            )

    def test_batched_vllm_rejects_prompt_that_exceeds_remaining_context(self, monkeypatch):
        class FakeTokenizer:
            def __call__(self, texts, add_special_tokens=False):
                assert add_special_tokens is False
                return {"input_ids": [[0] * 700 for _ in texts]}

        session = BatchedVllmServeSession(VLLMConfig(model="gs://bucket/model", max_model_len=4096))
        session._env = types.SimpleNamespace(model_id="fake-model", server_url="http://127.0.0.1:8000/v1")
        session._tokenizer = FakeTokenizer()

        def fail_post(*args, **kwargs):
            raise AssertionError("generate_from_prompt_texts should fail before making an HTTP request")

        monkeypatch.setattr("marin.alignment.batched_vllm_serve.requests.post", fail_post)

        with pytest.raises(ValueError, match="exceeds the model context window"):
            session.generate_from_prompt_texts(
                ["prompt"],
                stage_name="understanding",
                temperature=0.0,
                max_tokens=3500,
                n=1,
            )

    def test_batched_vllm_rejects_render_messages_for_gpt_oss(self):
        session = BatchedVllmServeSession(
            VLLMConfig(
                model="gs://bucket/unsloth--gpt-oss-20b-BF16-vllm",
                hf_overrides={"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss"},
            )
        )

        with pytest.raises(ValueError, match="must use generate_from_messages\\(\\) -> /v1/chat/completions"):
            session.render_messages([[{"role": "user", "content": "hello"}]])

    def test_batched_vllm_rejects_prompt_texts_for_gpt_oss(self):
        session = BatchedVllmServeSession(
            VLLMConfig(
                model="gs://bucket/unsloth--gpt-oss-20b-BF16-vllm",
                hf_overrides={"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss"},
            )
        )

        with pytest.raises(ValueError, match="must use generate_from_messages\\(\\) -> /v1/chat/completions"):
            session.generate_from_prompt_texts(
                ["prompt"],
                stage_name="understanding",
                temperature=0.0,
                max_tokens=2048,
                n=1,
            )

    def test_batched_vllm_uses_chat_completions_for_gpt_oss(self, monkeypatch):
        session = BatchedVllmServeSession(
            VLLMConfig(
                model="gs://bucket/unsloth--gpt-oss-20b-BF16-vllm",
                hf_overrides={"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss"},
            )
        )
        session._env = types.SimpleNamespace(
            model_id="gs://bucket/unsloth--gpt-oss-20b-BF16-vllm",
            server_url="http://127.0.0.1:8000/v1",
        )

        calls: list[tuple[str, dict[str, object], int]] = []
        payloads = [
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "first response", "reasoning_content": "reasoning one"},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "second response", "reasoning_content": "reasoning two"},
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 21},
            },
        ]

        def fake_post(url, json, timeout):
            calls.append((url, json, timeout))
            return self._FakeRequestsResponse(payloads[len(calls) - 1])

        monkeypatch.setattr("marin.alignment.batched_vllm_serve.requests.post", fake_post)

        outputs = session.generate_from_messages(
            [
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}],
            ],
            stage_name="understanding",
            temperature=0.0,
            max_tokens=2048,
            n=1,
        )

        assert outputs == [["first response"], ["second response"]]
        # Calls are sent concurrently, so order may vary — check both are present
        assert len(calls) == 2
        call_messages = sorted([c[1]["messages"][0]["content"] for c in calls])
        assert call_messages == ["first", "second"]
        for url, body, timeout in calls:
            assert url == "http://127.0.0.1:8000/v1/chat/completions"
            assert body["model"] == "gs://bucket/unsloth--gpt-oss-20b-BF16-vllm"
            assert body["reasoning_effort"] == "low"
            assert body["temperature"] == 0.0
            assert body["max_tokens"] == 2048
            assert timeout == 900
        metrics = session.metrics_snapshot()
        # Concurrent batch records one aggregated metric
        assert metrics["totals"]["request_count"] == 1
        assert metrics["totals"]["request_prompt_count"] == 2
        assert metrics["totals"]["input_token_count"] == 21
        assert metrics["totals"]["output_token_count"] == 41

    def test_batched_vllm_inserts_empty_result_on_non_stop_gpt_oss_chat_response(self, monkeypatch, caplog):
        session = BatchedVllmServeSession(
            VLLMConfig(
                model="gs://bucket/unsloth--gpt-oss-20b-BF16-vllm",
                hf_overrides={"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss"},
            )
        )
        session._env = types.SimpleNamespace(
            model_id="gs://bucket/unsloth--gpt-oss-20b-BF16-vllm",
            server_url="http://127.0.0.1:8000/v1",
        )

        def fake_post(url, json, timeout):
            return self._FakeRequestsResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "partial", "reasoning_content": "reasoning"},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 100},
                }
            )

        monkeypatch.setattr("marin.alignment.batched_vllm_serve.requests.post", fake_post)

        outputs = session.generate_from_messages(
            [[{"role": "user", "content": "first"}]],
            stage_name="understanding",
            temperature=0.0,
            max_tokens=2048,
            n=1,
        )

        assert outputs == [[]]
        assert "Expected finish_reason='stop', got 'length'" in caplog.text
        assert "inserting empty results for failed items" in caplog.text


class TestResponseHelpers:
    def test_build_chosen_messages_with_spec_guidance(self, sample_prompt):
        behavior_statements = {"be_helpful": "Always be helpful and friendly."}
        messages = _build_chosen_messages(sample_prompt, behavior_statements)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "customer service agent" in messages[0]["content"]
        assert "Always be helpful" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_build_rejected_messages_without_spec_guidance(self, sample_prompt):
        messages = _build_rejected_messages(sample_prompt, RejectedPromptStrategy.UNGUIDED, None)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "behavioral guideline" not in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_build_rejected_messages_no_system_prompt(self):
        prompt = {"behavior_id": "x", "system_prompt": "", "user_message": "Hello"}
        messages = _build_rejected_messages(prompt, RejectedPromptStrategy.UNGUIDED, None)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_build_rejected_messages_opposite_mode(self, sample_prompt):
        behavior_statements = {"be_helpful": "Always be helpful and friendly."}
        messages = _build_rejected_messages(sample_prompt, RejectedPromptStrategy.OPPOSITE, behavior_statements)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "PRIMARY DIRECTIVE" in messages[0]["content"]
        assert "Deliberately violate" in messages[0]["content"]
        assert "Always be helpful and friendly." in messages[0]["content"]
        assert "customer service agent" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_response_gen_config_rejects_chosen_with_rejected_strategy(self):
        with pytest.raises(ValueError, match="Chosen responses cannot specify rejected_prompt_strategy"):
            ResponseGenConfig(
                prompts_path="gs://bucket/prompts",
                output_path="gs://bucket/output",
                model_config=OpenAIConfig(model="gpt-4.1"),
                role=ResponseRole.CHOSEN,
                rejected_prompt_strategy=RejectedPromptStrategy.OPPOSITE,
                behavior_statements_path="gs://bucket/spec.jsonl",
            )

    def test_response_gen_config_rejects_rejected_opposite_without_spec(self):
        with pytest.raises(ValueError, match="Rejected opposite responses require behavior_statements_path"):
            ResponseGenConfig(
                prompts_path="gs://bucket/prompts",
                output_path="gs://bucket/output",
                model_config=OpenAIConfig(model="gpt-4.1"),
                role=ResponseRole.REJECTED,
                rejected_prompt_strategy=RejectedPromptStrategy.OPPOSITE,
            )

    def test_response_gen_config_rejects_rejected_unguided_with_spec(self):
        with pytest.raises(ValueError, match="Rejected unguided responses must not receive behavior_statements_path"):
            ResponseGenConfig(
                prompts_path="gs://bucket/prompts",
                output_path="gs://bucket/output",
                model_config=OpenAIConfig(model="gpt-4.1"),
                role=ResponseRole.REJECTED,
                rejected_prompt_strategy=RejectedPromptStrategy.UNGUIDED,
                behavior_statements_path="gs://bucket/spec.jsonl",
            )

    def test_response_pair_gen_config_rejects_opposite_without_rejected_spec(self, sample_spec_jsonl):
        with pytest.raises(ValueError, match="Rejected opposite responses require rejected_behavior_statements_path"):
            ResponsePairGenConfig(
                prompts_path="gs://bucket/prompts",
                chosen_output_path="gs://bucket/output/chosen",
                rejected_output_path="gs://bucket/output/rejected",
                chosen_model_config=VLLMConfig(model="gs://bucket/model", tensor_parallel_size=4, tpu_type="v5p-8"),
                rejected_model_config=VLLMConfig(model="gs://bucket/model", tensor_parallel_size=4, tpu_type="v5p-8"),
                chosen_behavior_statements_path=str(sample_spec_jsonl),
                rejected_prompt_strategy=RejectedPromptStrategy.OPPOSITE,
            )

    @patch("marin.alignment.generate_responses.write_vllm_metrics_artifact")
    @patch("marin.alignment.generate_responses.write_sharded_jsonl_gz")
    @patch("marin.alignment.generate_responses.load_sharded_jsonl_gz")
    @patch("marin.alignment.generate_responses.BatchedVllmServeSession")
    def test_generate_responses_vllm_uses_batched_serve(
        self,
        mock_session_cls,
        mock_load_shards,
        mock_write_shards,
        mock_write_metrics,
        sample_prompt,
    ):
        mock_load_shards.return_value = [sample_prompt]
        mock_session = mock_session_cls.return_value.__enter__.return_value
        mock_session.generate_from_messages.return_value = [["Refactored response"]]
        mock_session.metrics_snapshot.return_value = {"stages": {"rejected": {}}, "totals": {}}

        config = ResponseGenConfig(
            prompts_path="gs://bucket/prompts",
            output_path="gs://bucket/output",
            model_config=VLLMConfig(model="gs://bucket/model", tensor_parallel_size=4, tpu_type="v5p-8"),
            role=ResponseRole.REJECTED,
            rejected_prompt_strategy=RejectedPromptStrategy.UNGUIDED,
            n=1,
            temperature=0.7,
            max_tokens=512,
        )

        generate_responses(config)

        mock_session_cls.assert_called_once()
        mock_session.generate_from_messages.assert_called_once()
        call_kwargs = mock_session.generate_from_messages.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["n"] == 1

        write_args = mock_write_shards.call_args.args
        assert write_args[1] == "gs://bucket/output"
        assert write_args[0] == [
            {
                "prompt_id": "be_helpful/cfg_000",
                "system_prompt": "You are a customer service agent.",
                "user_message": "I need help with my order.",
                "behavior_id": "be_helpful",
                "rubric": "GOOD: Helpful response. BAD: Dismissive.",
                "model": "gs://bucket/model",
                "response_role": "rejected",
                "behavior_prompt_mode_resolved": "unguided",
                "rejected_prompt_strategy": "unguided",
                "responses": [{"content": "Refactored response", "index": 0}],
            }
        ]
        mock_write_metrics.assert_called_once()
        assert mock_write_metrics.call_args.kwargs["logical_stage"] == "response_generation"

    @patch("marin.alignment.generate_responses.write_vllm_metrics_artifact")
    @patch("marin.alignment.generate_responses.write_sharded_jsonl_gz")
    @patch("marin.alignment.generate_responses.load_sharded_jsonl_gz")
    @patch("marin.alignment.generate_responses.BatchedVllmServeSession")
    def test_generate_responses_vllm_microbatches_prompts_for_live_progress(
        self,
        mock_session_cls,
        mock_load_shards,
        mock_write_shards,
        mock_write_metrics,
        sample_prompt,
    ):
        prompt_one = dict(sample_prompt)
        prompt_two = dict(sample_prompt, config_id="cfg_001", user_message="Need help with billing.")
        prompt_three = dict(sample_prompt, config_id="cfg_002", user_message="Need help with shipping.")
        mock_load_shards.return_value = [prompt_one, prompt_two, prompt_three]
        mock_session = mock_session_cls.return_value.__enter__.return_value
        mock_session.generate_from_messages.side_effect = [
            [["Response 0"], ["Response 1"]],
            [["Response 2"]],
        ]
        mock_session.metrics_snapshot.return_value = {"stages": {"rejected": {}}, "totals": {}}

        config = ResponseGenConfig(
            prompts_path="gs://bucket/prompts",
            output_path="gs://bucket/output",
            model_config=VLLMConfig(model="gs://bucket/model", tensor_parallel_size=4, tpu_type="v5p-8"),
            role=ResponseRole.REJECTED,
            rejected_prompt_strategy=RejectedPromptStrategy.UNGUIDED,
            n=1,
            temperature=0.7,
            max_tokens=512,
            local_serve_batch_size=2,
        )

        generate_responses(config)

        assert mock_session.generate_from_messages.call_count == 2
        first_batch = mock_session.generate_from_messages.call_args_list[0].args[0]
        second_batch = mock_session.generate_from_messages.call_args_list[1].args[0]
        assert len(first_batch) == 2
        assert len(second_batch) == 1

        written_records = mock_write_shards.call_args.args[0]
        assert [record["prompt_id"] for record in written_records] == [
            "be_helpful/cfg_000",
            "be_helpful/cfg_001",
            "be_helpful/cfg_002",
        ]
        assert [record["responses"][0]["content"] for record in written_records] == [
            "Response 0",
            "Response 1",
            "Response 2",
        ]
        mock_write_metrics.assert_called_once()

    @patch("marin.alignment.generate_responses.write_vllm_metrics_artifact")
    @patch("marin.alignment.generate_responses.write_sharded_jsonl_gz")
    @patch("marin.alignment.generate_responses.load_sharded_jsonl_gz")
    @patch("marin.alignment.generate_responses.BatchedVllmServeSession")
    def test_generate_response_pair_reuses_session_for_same_local_model(
        self,
        mock_session_cls,
        mock_load_shards,
        mock_write_shards,
        mock_write_metrics,
        sample_prompt,
        sample_spec_jsonl,
    ):
        mock_load_shards.return_value = [sample_prompt]
        mock_session = mock_session_cls.return_value.__enter__.return_value
        mock_session.generate_from_messages.side_effect = [
            [["Chosen response"]],
            [["Rejected response 1", "Rejected response 2"]],
        ]
        mock_session.metrics_snapshot.return_value = {"stages": {"chosen": {}, "rejected": {}}, "totals": {}}

        model_config = VLLMConfig(model="gs://bucket/model", tensor_parallel_size=4, tpu_type="v5p-8")
        config = ResponsePairGenConfig(
            prompts_path="gs://bucket/prompts",
            chosen_output_path="gs://bucket/output/chosen",
            rejected_output_path="gs://bucket/output/rejected",
            chosen_model_config=model_config,
            rejected_model_config=model_config,
            chosen_n=1,
            chosen_temperature=0.4,
            chosen_max_tokens=256,
            chosen_behavior_statements_path=str(sample_spec_jsonl),
            rejected_n=2,
            rejected_temperature=0.8,
            rejected_max_tokens=512,
            rejected_prompt_strategy=RejectedPromptStrategy.UNGUIDED,
        )

        generate_response_pair(config)

        mock_session_cls.assert_called_once_with(model_config)
        assert mock_session.generate_from_messages.call_count == 2
        chosen_call = mock_session.generate_from_messages.call_args_list[0].kwargs
        rejected_call = mock_session.generate_from_messages.call_args_list[1].kwargs
        assert chosen_call["temperature"] == 0.4
        assert chosen_call["max_tokens"] == 256
        assert chosen_call["n"] == 1
        assert rejected_call["temperature"] == 0.8
        assert rejected_call["max_tokens"] == 512
        assert rejected_call["n"] == 2
        assert mock_write_shards.call_count == 2
        assert mock_write_shards.call_args_list[0].args[1] == "gs://bucket/output/chosen"
        assert mock_write_shards.call_args_list[1].args[1] == "gs://bucket/output/rejected"
        mock_write_metrics.assert_called_once()
        assert mock_write_metrics.call_args.args[0] == "gs://bucket/output/artifacts/vllm_metrics.json"

    @patch("marin.alignment.generate_responses.write_vllm_metrics_artifact")
    @patch("marin.alignment.generate_responses.write_sharded_jsonl_gz")
    @patch("marin.alignment.generate_responses.load_sharded_jsonl_gz")
    @patch("marin.alignment.generate_responses.BatchedVllmServeSession")
    def test_generate_response_pair_uses_separate_sessions_for_different_local_models(
        self,
        mock_session_cls,
        mock_load_shards,
        mock_write_shards,
        mock_write_metrics,
        sample_prompt,
        sample_spec_jsonl,
    ):
        mock_load_shards.return_value = [sample_prompt]
        first_context = MagicMock()
        second_context = MagicMock()
        first_session = first_context.__enter__.return_value
        second_session = second_context.__enter__.return_value
        first_session.generate_from_messages.return_value = [["Chosen response"]]
        second_session.generate_from_messages.return_value = [["Rejected response"]]
        first_session.metrics_snapshot.return_value = {"stages": {"chosen": {}}, "totals": {}}
        second_session.metrics_snapshot.return_value = {"stages": {"rejected": {}}, "totals": {}}
        mock_session_cls.side_effect = [first_context, second_context]

        config = ResponsePairGenConfig(
            prompts_path="gs://bucket/prompts",
            chosen_output_path="gs://bucket/output/chosen",
            rejected_output_path="gs://bucket/output/rejected",
            chosen_model_config=VLLMConfig(model="gs://bucket/chosen", tensor_parallel_size=4, tpu_type="v5p-8"),
            rejected_model_config=VLLMConfig(
                model="gs://bucket/rejected",
                tensor_parallel_size=4,
                tpu_type="v5p-8",
            ),
            chosen_n=1,
            rejected_n=1,
            chosen_behavior_statements_path=str(sample_spec_jsonl),
            rejected_prompt_strategy=RejectedPromptStrategy.UNGUIDED,
        )

        generate_response_pair(config)

        assert mock_session_cls.call_count == 2
        assert first_session.generate_from_messages.call_count == 1
        assert second_session.generate_from_messages.call_count == 1
        assert mock_write_shards.call_count == 2
        mock_write_metrics.assert_called_once()
        assert mock_write_metrics.call_args.args[0] == "gs://bucket/output/artifacts/vllm_metrics.json"


# ===========================================================================
# Tests: judge.py — parse judge response
# ===========================================================================


class TestJudgeParsing:
    def test_parse_json_in_code_block(self):
        content = '```json\n{"score": 8, "confidence": 0.9, "explanation": "Good", "highlights": []}\n```'
        result = parse_judge_response(content)
        assert result["score"] == 8
        assert result["confidence"] == 0.9

    def test_parse_raw_json(self):
        content = '{"score": 3, "confidence": 0.5, "explanation": "Bad"}'
        result = parse_judge_response(content)
        assert result["score"] == 3

    def test_parse_json_with_surrounding_text(self):
        content = 'Here is my judgment:\n{"score": 7, "confidence": 0.8, "explanation": "OK"}\nThat is all.'
        result = parse_judge_response(content)
        assert result["score"] == 7

    def test_parse_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            parse_judge_response("not json at all")

    def test_parse_compliance_result_returns_none_on_parse_failure(self):
        """Parse failures must surface as score=None, not a coerced default.

        Coercing to 0 or 5 biases downstream mean-score and compliance-rate
        aggregates — parse failures mean 'unknown', not 'zero' or 'midpoint'.
        """
        result = parse_compliance_result("not json at all")
        assert result.score is None
        assert result.compliant is None
        assert "Parse failure" in result.explanation

    def test_parse_compliance_result_returns_none_when_score_missing(self):
        """Valid JSON without a 'score' field is also a parse failure."""
        result = parse_compliance_result('{"confidence": 0.9, "explanation": "missing score key"}')
        assert result.score is None
        assert result.compliant is None

    def test_compliance_result_from_judge_output_missing_score(self):
        """from_judge_output maps a missing 'score' key to None rather than a default."""
        result = ComplianceResult.from_judge_output({"confidence": 0.8, "explanation": "no score"})
        assert result.score is None
        assert result.compliant is None


class TestAlignResponseOrchestration:
    def test_align_uses_combined_response_step_for_same_local_teacher_and_rejected(self, sample_spec_jsonl):
        pretrained_step = ExecutorStep(
            name="pretrained",
            fn=remote(_noop_remote, resources=ResourceConfig.with_cpu(cpu=1, ram="4g", disk="4g")),
            config={},
        )
        teacher_model = VLLMConfig(model="gs://bucket/chosen", tensor_parallel_size=4, tpu_type="v5p-8")
        rejected_model = teacher_model

        steps = align(
            name="local-local",
            pretrained_model=pretrained_step,
            spec=str(sample_spec_jsonl),
            model_config=object(),
            teacher_model=teacher_model,
            rejected_model=rejected_model,
            align_config=AlignConfig(
                ideation_model="gpt-4.1",
                extract_model="gpt-4.1-mini",
                judge_model="gpt-4.1",
            ),
        )

        assert [step.name for step in steps[:5]] == [
            "align/local-local/spec",
            "align/local-local/prompts",
            "align/local-local/responses",
            "align/local-local/judgments",
            "align/local-local/preference_pairs",
        ]
        response_step = steps[2]
        judgments_step = steps[3]
        assert response_step.fn.resources.ram == "128g"
        assert response_step.fn.resources.disk == "50g"
        assert response_step.config.chosen_output_path.name == "chosen"
        assert response_step.config.rejected_output_path.name == "rejected"
        assert isinstance(judgments_step.config.chosen_responses_path, InputName)
        assert judgments_step.config.chosen_responses_path.step is response_step
        assert judgments_step.config.chosen_responses_path.name == "chosen"
        assert judgments_step.config.rejected_responses_path.step is response_step
        assert judgments_step.config.rejected_responses_path.name == "rejected"

    def test_align_preserves_input_name_model_path_in_combined_response_config(self, sample_spec_jsonl):
        pretrained_step = ExecutorStep(
            name="pretrained",
            fn=remote(_noop_remote, resources=ResourceConfig.with_cpu(cpu=1, ram="4g", disk="4g")),
            config={},
        )
        model_step = ExecutorStep(
            name="model",
            fn=remote(_noop_remote, resources=ResourceConfig.with_cpu(cpu=1, ram="4g", disk="4g")),
            config={},
        )
        shared_model = VLLMConfig(model=output_path_of(model_step), tensor_parallel_size=4, tpu_type="v5p-8")

        steps = align(
            name="same-model-input-name",
            pretrained_model=pretrained_step,
            spec=str(sample_spec_jsonl),
            model_config=object(),
            teacher_model=shared_model,
            rejected_model=shared_model,
            align_config=AlignConfig(
                ideation_model="gpt-4.1",
                extract_model="gpt-4.1-mini",
                judge_model="gpt-4.1",
            ),
        )

        response_step = steps[2]
        assert isinstance(response_step.config.chosen_model_config["model"], InputName)
        assert response_step.config.chosen_model_config["model"].step is model_step
        assert isinstance(response_step.config.rejected_model_config["model"], InputName)
        assert response_step.config.rejected_model_config["model"].step is model_step

    def test_align_auto_uses_parallel_separate_steps_for_different_local_models(self, sample_spec_jsonl):
        pretrained_step = ExecutorStep(
            name="pretrained",
            fn=remote(_noop_remote, resources=ResourceConfig.with_cpu(cpu=1, ram="4g", disk="4g")),
            config={},
        )
        teacher_model = VLLMConfig(
            model="gs://bucket/chosen",
            tensor_parallel_size=4,
            tpu_type="v5p-8",
            disk="50g",
            ram="128g",
        )
        rejected_model = VLLMConfig(
            model="gs://bucket/rejected",
            tensor_parallel_size=4,
            tpu_type="v5p-8",
            disk="75g",
            ram="256g",
        )

        steps = align(
            name="different-local",
            pretrained_model=pretrained_step,
            spec=str(sample_spec_jsonl),
            model_config=object(),
            teacher_model=teacher_model,
            rejected_model=rejected_model,
            align_config=AlignConfig(
                ideation_model="gpt-4.1",
                extract_model="gpt-4.1-mini",
                judge_model="gpt-4.1",
            ),
        )

        assert [step.name for step in steps[:6]] == [
            "align/different-local/spec",
            "align/different-local/prompts",
            "align/different-local/chosen",
            "align/different-local/rejected",
            "align/different-local/judgments",
            "align/different-local/preference_pairs",
        ]
        rejected_step = steps[3]
        assert rejected_step.config.dependency_path is None

    def test_align_parallel_keeps_separate_steps_without_dependency(self, sample_spec_jsonl):
        pretrained_step = ExecutorStep(
            name="pretrained",
            fn=remote(_noop_remote, resources=ResourceConfig.with_cpu(cpu=1, ram="4g", disk="4g")),
            config={},
        )
        teacher_model = VLLMConfig(model="gs://bucket/chosen", tensor_parallel_size=4, tpu_type="v5p-8")
        rejected_model = VLLMConfig(model="gs://bucket/rejected", tensor_parallel_size=4, tpu_type="v5p-8")

        steps = align(
            name="parallel-local",
            pretrained_model=pretrained_step,
            spec=str(sample_spec_jsonl),
            model_config=object(),
            teacher_model=teacher_model,
            rejected_model=rejected_model,
            align_config=AlignConfig(
                ideation_model="gpt-4.1",
                extract_model="gpt-4.1-mini",
                judge_model="gpt-4.1",
                response_execution_mode=ResponseExecutionMode.PARALLEL,
            ),
        )

        assert [step.name for step in steps[:6]] == [
            "align/parallel-local/spec",
            "align/parallel-local/prompts",
            "align/parallel-local/chosen",
            "align/parallel-local/rejected",
            "align/parallel-local/judgments",
            "align/parallel-local/preference_pairs",
        ]
        assert steps[3].config.dependency_path is None

    def test_align_serializes_different_local_teacher_and_rejected_steps(self, sample_spec_jsonl):
        pretrained_step = ExecutorStep(
            name="pretrained",
            fn=remote(_noop_remote, resources=ResourceConfig.with_cpu(cpu=1, ram="4g", disk="4g")),
            config={},
        )
        teacher_model = VLLMConfig(
            model="gs://bucket/chosen",
            tensor_parallel_size=4,
            tpu_type="v5p-8",
            disk="50g",
            ram="128g",
        )
        rejected_model = VLLMConfig(
            model="gs://bucket/rejected",
            tensor_parallel_size=4,
            tpu_type="v5p-8",
            disk="75g",
            ram="256g",
        )

        steps = align(
            name="serialized-local",
            pretrained_model=pretrained_step,
            spec=str(sample_spec_jsonl),
            model_config=object(),
            teacher_model=teacher_model,
            rejected_model=rejected_model,
            align_config=AlignConfig(
                ideation_model="gpt-4.1",
                extract_model="gpt-4.1-mini",
                judge_model="gpt-4.1",
                response_execution_mode=ResponseExecutionMode.SERIALIZED,
            ),
        )

        assert [step.name for step in steps[:6]] == [
            "align/serialized-local/spec",
            "align/serialized-local/prompts",
            "align/serialized-local/chosen",
            "align/serialized-local/rejected",
            "align/serialized-local/judgments",
            "align/serialized-local/preference_pairs",
        ]
        chosen_step = steps[2]
        rejected_step = steps[3]
        assert rejected_step.config.dependency_path.step is chosen_step
        assert rejected_step.config.dependency_path.name is None

    def test_align_reuse_same_model_requires_matching_local_models(self, sample_spec_jsonl):
        pretrained_step = ExecutorStep(
            name="pretrained",
            fn=remote(_noop_remote, resources=ResourceConfig.with_cpu(cpu=1, ram="4g", disk="4g")),
            config={},
        )
        with pytest.raises(ValueError, match="reuse_same_model"):
            align(
                name="invalid-reuse",
                pretrained_model=pretrained_step,
                spec=str(sample_spec_jsonl),
                model_config=object(),
                teacher_model=VLLMConfig(model="gs://bucket/chosen", tensor_parallel_size=4, tpu_type="v5p-8"),
                rejected_model=VLLMConfig(model="gs://bucket/rejected", tensor_parallel_size=4, tpu_type="v5p-8"),
                align_config=AlignConfig(
                    ideation_model="gpt-4.1",
                    extract_model="gpt-4.1-mini",
                    judge_model="gpt-4.1",
                    response_execution_mode=ResponseExecutionMode.REUSE_SAME_MODEL,
                ),
            )

    def test_align_opposite_strategy_sets_rejected_config_only(self, sample_spec_jsonl):
        pretrained_step = ExecutorStep(
            name="pretrained",
            fn=remote(_noop_remote, resources=ResourceConfig.with_cpu(cpu=1, ram="4g", disk="4g")),
            config={},
        )
        teacher_model = VLLMConfig(model="gs://bucket/model", tensor_parallel_size=4, tpu_type="v5p-8")

        steps = align(
            name="opposite-local",
            pretrained_model=pretrained_step,
            spec=str(sample_spec_jsonl),
            model_config=object(),
            teacher_model=teacher_model,
            rejected_model=teacher_model,
            align_config=AlignConfig(
                ideation_model="gpt-4.1",
                extract_model="gpt-4.1-mini",
                judge_model="gpt-4.1",
                response_execution_mode=ResponseExecutionMode.REUSE_SAME_MODEL,
                rejected_prompt_strategy=RejectedPromptStrategy.OPPOSITE,
            ),
        )

        responses_step = steps[2]
        assert responses_step.name == "align/opposite-local/responses"
        assert responses_step.config.chosen_behavior_statements_path.step.name == "align/opposite-local/spec"
        assert responses_step.config.rejected_prompt_strategy == RejectedPromptStrategy.OPPOSITE
        assert responses_step.config.rejected_behavior_statements_path.step.name == "align/opposite-local/spec"

    def test_align_keeps_separate_response_steps_when_only_teacher_is_local(self, sample_spec_jsonl):
        pretrained_step = ExecutorStep(
            name="pretrained",
            fn=remote(_noop_remote, resources=ResourceConfig.with_cpu(cpu=1, ram="4g", disk="4g")),
            config={},
        )
        teacher_model = VLLMConfig(model="gs://bucket/chosen", tensor_parallel_size=4, tpu_type="v5p-8")
        rejected_model = OpenAIConfig(model="gpt-4.1-mini")

        steps = align(
            name="mixed",
            pretrained_model=pretrained_step,
            spec=str(sample_spec_jsonl),
            model_config=object(),
            teacher_model=teacher_model,
            rejected_model=rejected_model,
            align_config=AlignConfig(
                ideation_model="gpt-4.1",
                extract_model="gpt-4.1-mini",
                judge_model="gpt-4.1",
            ),
        )

        assert [step.name for step in steps[:6]] == [
            "align/mixed/spec",
            "align/mixed/prompts",
            "align/mixed/chosen",
            "align/mixed/rejected",
            "align/mixed/judgments",
            "align/mixed/preference_pairs",
        ]


# ===========================================================================
# Tests: llm_client.py
# ===========================================================================


class TestLLMClient:
    @patch("marin.alignment.llm_client.OpenAI")
    def test_llm_chat_returns_responses(self, mock_openai, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response("Hello world!")
        mock_openai.return_value = mock_client
        responses = llm_chat(
            config=OpenAIConfig(model="gpt-4.1"),
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert len(responses) == 1
        assert responses[0].content == "Hello world!"
        assert responses[0].model == "gpt-4.1"
        mock_client.chat.completions.create.assert_called_once()

    @patch("marin.alignment.llm_client.OpenAI")
    def test_llm_chat_with_string_config(self, mock_openai, mock_openai_response):
        """Bare string is auto-converted to OpenAIConfig."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response("Response")
        mock_openai.return_value = mock_client
        responses = llm_chat(
            config="gpt-4.1",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert len(responses) == 1
        assert responses[0].content == "Response"

    @patch("marin.alignment.llm_client.OpenAI")
    def test_llm_chat_with_system_prompt(self, mock_openai, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response("Response")
        mock_openai.return_value = mock_client
        llm_chat(
            config=OpenAIConfig(model="gpt-4.1"),
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="You are helpful.",
        )
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"

    @patch("marin.alignment.llm_client.OpenAI")
    def test_llm_chat_single(self, mock_openai, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response("Single response")
        mock_openai.return_value = mock_client
        response = llm_chat_single(
            config=OpenAIConfig(model="gpt-4.1"),
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert isinstance(response, LLMResponse)
        assert response.content == "Single response"

    @patch("marin.alignment.llm_client.OpenAI")
    def test_llm_chat_multiple_n(self, mock_openai):
        mock_choice1 = MagicMock()
        mock_choice1.message.content = "Response 1"
        mock_choice2 = MagicMock()
        mock_choice2.message.content = "Response 2"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice1, mock_choice2]
        mock_response.model = "gpt-4.1"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        responses = llm_chat(
            config=OpenAIConfig(model="gpt-4.1"),
            messages=[{"role": "user", "content": "Hi"}],
            n=2,
        )
        assert len(responses) == 2
        assert responses[0].content == "Response 1"
        assert responses[1].content == "Response 2"

    def test_get_or_create_vllm_engine_defaults_remote_models_to_runai_streamer(self, monkeypatch):
        captured_kwargs = {}

        class FakeLLM:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

            def get_tokenizer(self):
                return MagicMock()

        monkeypatch.setattr(llm_client_module, "_vllm_engine_cache", {})
        monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM))

        llm_client_module.get_or_create_vllm_engine(VLLMConfig(model="gs://bucket/model"))

        assert captured_kwargs["load_format"] == "runai_streamer"

    def test_get_or_create_vllm_engine_preserves_explicit_load_format(self, monkeypatch):
        captured_kwargs = {}

        class FakeLLM:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

            def get_tokenizer(self):
                return MagicMock()

        monkeypatch.setattr(llm_client_module, "_vllm_engine_cache", {})
        monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM))

        llm_client_module.get_or_create_vllm_engine(
            VLLMConfig(model="gs://bucket/model", load_format="runai_streamer_sharded")
        )

        assert captured_kwargs["load_format"] == "runai_streamer_sharded"

    def test_get_or_create_vllm_engine_passes_explicit_tokenizer(self, monkeypatch):
        captured_kwargs = {}

        class FakeLLM:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

            def get_tokenizer(self):
                return MagicMock()

        monkeypatch.setattr(llm_client_module, "_vllm_engine_cache", {})
        monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM))

        llm_client_module.get_or_create_vllm_engine(
            VLLMConfig(model="gs://bucket/model/original", tokenizer="gs://bucket/model")
        )

        assert captured_kwargs["tokenizer"] == "gs://bucket/model"

    def test_get_or_create_vllm_engine_passes_hf_overrides(self, monkeypatch):
        captured_kwargs = {}

        class FakeLLM:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

            def get_tokenizer(self):
                return MagicMock()

        monkeypatch.setattr(llm_client_module, "_vllm_engine_cache", {})
        monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM))

        llm_client_module.get_or_create_vllm_engine(
            VLLMConfig(
                model="gs://bucket/model/original",
                hf_overrides={"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss"},
            )
        )

        assert captured_kwargs["hf_overrides"] == {
            "architectures": ["GptOssForCausalLM"],
            "model_type": "gpt_oss",
        }

    def test_get_or_create_vllm_engine_rejects_docker_backed_configs(self):
        with pytest.raises(ValueError, match="docker-backed VLLMConfig"):
            llm_client_module.get_or_create_vllm_engine(
                VLLMConfig(model="gs://bucket/model", gpu_type="H100", tpu_type=None)
            )


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
            config_id="cfg_000",
            axis_config={"complexity": "simple", "tone": "casual"},
            axes_metadata=[
                {"axis": "complexity", "spectrum": ["simple", "complex"], "description": "Task complexity"},
                {"axis": "tone", "spectrum": ["casual", "formal"], "description": "Response tone"},
            ],
        )
        assert "evaluation scenario designer" in system_prompt
        assert "helpfulness" in user_prompt
        assert "Configuration ID: cfg_000" in user_prompt
        assert "simple" in user_prompt
        assert "<scenario>" in user_prompt

    def test_extraction_prompt_with_system(self):
        system_prompt, user_prompt = make_extraction_prompt(
            scenario={"description": "A user asks about weather."},
            include_system_prompt=True,
        )
        assert "system_prompt" in system_prompt
        assert "user_message" in system_prompt
        assert "A user asks about weather." in user_prompt

    def test_extraction_prompt_without_system(self):
        system_prompt, _ = make_extraction_prompt(
            scenario={"description": "A user asks about weather."},
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

    @patch("marin.alignment.generate_prompts.compute_coverage_stats", return_value={"covered": 1})
    @patch(
        "marin.alignment.generate_prompts.generate_covering_configs",
        return_value=[{"request_complexity": "simple", "domain": "general"}],
    )
    @patch("marin.alignment.generate_prompts.llm_chat_single")
    def test_full_pipeline(
        self,
        mock_llm,
        _mock_covering_configs,
        _mock_coverage_stats,
        sample_spec_jsonl,
        tmp_path,
    ):
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
            model="gpt-4.1",
        )

        # Stage 2: Concretize response
        stage2_response = LLMResponse(
            content=(
                "<scenario>A student asks for help with basic math homework. "
                '"You are a math tutor." The student says "Can you help me with fractions?"</scenario>\n'
                "<rubric>GOOD: Patient explanation. BAD: Dismissive.</rubric>\n"
            ),
            model="gpt-4.1",
        )

        # Dispatch mock based on system_prompt content:
        # - Stage 1: understanding system prompt contains "alignment research assistant"
        # - Stage 2: concretize system prompt contains "scenario designer"
        # - Stage 3: extraction — returns <scenario_N> with correct index

        def mock_side_effect(**kwargs):
            sys_prompt = kwargs.get("system_prompt") or ""

            if "alignment research assistant" in sys_prompt:
                return stage1_response
            elif "scenario designer" in sys_prompt:
                return stage2_response
            else:
                return LLMResponse(
                    content=(
                        "<system_prompt>You are a math tutor.</system_prompt>\n"
                        "<user_message>Can you help me with fractions?</user_message>\n"
                    ),
                    model="gpt-4.1-mini",
                )

        mock_llm.side_effect = mock_side_effect

        output_path = str(tmp_path / "prompts_output")

        config = PromptGenConfig(
            spec_path=str(sample_spec_jsonl),
            output_path=output_path,
            ideation_model="gpt-4.1",
            extract_model="gpt-4.1-mini",
            covering_strength=2,
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

    @patch("marin.alignment.generate_prompts.compute_coverage_stats", return_value={"covered": 5})
    @patch(
        "marin.alignment.generate_prompts.generate_covering_configs",
        return_value=[
            {"request_complexity": "simple", "domain": "general"},
            {"request_complexity": "moderate", "domain": "general"},
            {"request_complexity": "complex", "domain": "general"},
            {"request_complexity": "simple", "domain": "technical"},
            {"request_complexity": "moderate", "domain": "technical"},
        ],
    )
    def test_concretization_stage_local_logs_progress_per_batch(
        self,
        _mock_covering_configs,
        _mock_coverage_stats,
        caplog,
        tmp_path,
    ):
        class FakeSession:
            def generate_from_messages(self, messages, **_kwargs):
                return [
                    [f"<scenario>Scenario {index}</scenario><rubric>Rubric {index}</rubric>"]
                    for index, _message in enumerate(messages)
                ]

        config = PromptGenConfig(
            spec_path=str(tmp_path / "unused.jsonl"),
            output_path=str(tmp_path / "output"),
            ideation_model="gpt-4.1",
            extract_model="gpt-4.1",
            local_serve_batch_size=2,
            concretize_max_attempts=1,
        )
        understandings = {
            "be_helpful": {
                "variation_axes": [
                    {
                        "axis": "request_complexity",
                        "spectrum": ["simple", "moderate", "complex"],
                        "description": "How hard the request is.",
                    },
                    {
                        "axis": "domain",
                        "spectrum": ["general", "technical"],
                        "description": "What topic domain the request belongs to.",
                    },
                ],
                "understanding": "Help the user.",
                "scientific_motivation": "Measure helpfulness.",
                "transcript_analyses": [],
            }
        }

        with caplog.at_level(logging.INFO):
            result = _run_concretization_stage_local(understandings, config, FakeSession())

        assert len(result["be_helpful"]["variations"]) == 5
        progress_messages = [record.message for record in caplog.records if "Stage 2 progress:" in record.message]
        assert any("2/5" in message for message in progress_messages)
        assert any("4/5" in message for message in progress_messages)
        assert any("5/5" in message for message in progress_messages)

    @patch("marin.alignment.generate_prompts.compute_coverage_stats", return_value={"covered": 1})
    @patch(
        "marin.alignment.generate_prompts.generate_covering_configs",
        return_value=[{"request_complexity": "simple"}],
    )
    @patch("marin.alignment.generate_prompts.BatchedVllmServeSession")
    def test_local_pipeline_reuses_single_session_when_models_match(
        self,
        mock_session_cls,
        _mock_covering_configs,
        _mock_coverage_stats,
        sample_spec_jsonl,
        tmp_path,
    ):
        class FakeSession:
            def __init__(self, outputs):
                self.outputs = list(outputs)
                self.batch_sizes: list[int] = []
                self.stage_names: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def generate_from_messages(self, message_batches, *, stage_name, temperature, max_tokens, n):
                self.batch_sizes.append(len(message_batches))
                self.stage_names.append(stage_name)
                next_outputs = self.outputs.pop(0)
                return [[text] for text in next_outputs]

            def metrics_snapshot(self):
                return {
                    "backend": "vllm_serve",
                    "model": "gs://bucket/shared",
                    "tensor_parallel_size": 4,
                    "max_model_len": 4096,
                    "tokenizer_load_seconds": 0.0,
                    "server_start_seconds": 0.0,
                    "session_enter_seconds": 0.0,
                    "totals": {},
                    "stages": {stage_name: {} for stage_name in self.stage_names},
                }

        session_instances: list[FakeSession] = []
        stage1_content = (
            "<behavior_understanding>Being helpful means assisting users.</behavior_understanding>\n"
            "<scientific_motivation>Testing helpfulness is important.</scientific_motivation>\n"
            "<variation_axes>\n"
            '[{"axis": "request_complexity", "spectrum": ["simple", "complex"], "description": "Complexity"}]\n'
            "</variation_axes>"
        )
        stage2_content = (
            "<scenario>A student asks for help with fractions.</scenario>\n"
            "<rubric>GOOD: Explain clearly. BAD: Refuse to help.</rubric>\n"
        )
        stage3_content = (
            "<system_prompt>You are a math tutor.</system_prompt>\n"
            "<user_message>Can you help me with fractions?</user_message>\n"
        )

        def session_factory(_config):
            session = FakeSession(
                outputs=[
                    [stage1_content],
                    [stage2_content],
                    [stage3_content],
                ]
            )
            session_instances.append(session)
            return session

        mock_session_cls.side_effect = session_factory

        shared_model = VLLMConfig(model="gs://bucket/shared", tensor_parallel_size=4, tpu_type="v5p-8")
        output_path = str(tmp_path / "prompt_local_shared")

        from marin.alignment.generate_prompts import generate_prompts_from_spec

        generate_prompts_from_spec(
            PromptGenConfig(
                spec_path=str(sample_spec_jsonl),
                output_path=output_path,
                ideation_model=shared_model,
                extract_model=shared_model,
                covering_strength=2,
                local_serve_batch_size=4,
                statement_ids=["be_helpful"],
            )
        )

        assert mock_session_cls.call_count == 1
        assert session_instances[0].batch_sizes == [1, 1, 1]
        assert session_instances[0].stage_names == ["understanding", "concretize", "extract"]
        prompts = load_sharded_jsonl_gz(output_path)
        assert len(prompts) == 1
        assert prompts[0]["behavior_id"] == "be_helpful"
        metrics_path = Path(output_path) / "artifacts" / "vllm_metrics.json"
        assert metrics_path.exists()
        metrics_payload = json.loads(metrics_path.read_text())
        assert metrics_payload["logical_stage"] == "prompt_generation"

    @patch("marin.alignment.generate_prompts.compute_coverage_stats", return_value={"covered": 1})
    @patch(
        "marin.alignment.generate_prompts.generate_covering_configs",
        return_value=[{"request_complexity": "simple"}],
    )
    @patch("marin.alignment.generate_prompts.BatchedVllmServeSession")
    def test_local_pipeline_switches_sessions_when_models_differ(
        self,
        mock_session_cls,
        _mock_covering_configs,
        _mock_coverage_stats,
        sample_spec_jsonl,
        tmp_path,
    ):
        class FakeSession:
            def __init__(self, outputs):
                self.outputs = list(outputs)
                self.batch_sizes: list[int] = []
                self.stage_names: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def generate_from_messages(self, message_batches, *, stage_name, temperature, max_tokens, n):
                self.batch_sizes.append(len(message_batches))
                self.stage_names.append(stage_name)
                next_outputs = self.outputs.pop(0)
                return [[text] for text in next_outputs]

            def metrics_snapshot(self):
                return {
                    "backend": "vllm_serve",
                    "model": "gs://bucket/fake",
                    "tensor_parallel_size": 4,
                    "max_model_len": 4096,
                    "tokenizer_load_seconds": 0.0,
                    "server_start_seconds": 0.0,
                    "session_enter_seconds": 0.0,
                    "totals": {},
                    "stages": {stage_name: {} for stage_name in self.stage_names},
                }

        session_instances: list[FakeSession] = []
        stage1_content = (
            "<behavior_understanding>Being helpful means assisting users.</behavior_understanding>\n"
            "<scientific_motivation>Testing helpfulness is important.</scientific_motivation>\n"
            "<variation_axes>\n"
            '[{"axis": "request_complexity", "spectrum": ["simple", "complex"], "description": "Complexity"}]\n'
            "</variation_axes>"
        )
        stage2_content = (
            "<scenario>A student asks for help with fractions.</scenario>\n"
            "<rubric>GOOD: Explain clearly. BAD: Refuse to help.</rubric>\n"
        )
        stage3_content = (
            "<system_prompt>You are a math tutor.</system_prompt>\n"
            "<user_message>Can you help me with fractions?</user_message>\n"
        )

        def session_factory(config):
            outputs = [[stage1_content], [stage2_content]] if len(session_instances) == 0 else [[stage3_content]]
            session = FakeSession(outputs=outputs)
            session_instances.append(session)
            return session

        mock_session_cls.side_effect = session_factory

        ideation_model = VLLMConfig(model="gs://bucket/ideation", tensor_parallel_size=4, tpu_type="v5p-8")
        extract_model = VLLMConfig(model="gs://bucket/extract", tensor_parallel_size=4, tpu_type="v5p-8")
        output_path = str(tmp_path / "prompt_local_split")

        from marin.alignment.generate_prompts import generate_prompts_from_spec

        generate_prompts_from_spec(
            PromptGenConfig(
                spec_path=str(sample_spec_jsonl),
                output_path=output_path,
                ideation_model=ideation_model,
                extract_model=extract_model,
                covering_strength=2,
                local_serve_batch_size=4,
                statement_ids=["be_helpful"],
            )
        )

        assert mock_session_cls.call_count == 2
        assert session_instances[0].batch_sizes == [1, 1]
        assert session_instances[1].batch_sizes == [1]
        metrics_payload = json.loads((Path(output_path) / "artifacts" / "vllm_metrics.json").read_text())
        session_names = {session["session_name"] for session in metrics_payload["sessions"]}
        assert session_names == {"ideation", "extract"}

    @patch("marin.alignment.generate_prompts.compute_coverage_stats", return_value={"covered": 2})
    @patch(
        "marin.alignment.generate_prompts.generate_covering_configs",
        return_value=[{"request_complexity": "simple"}],
    )
    @patch("marin.alignment.generate_prompts.BatchedVllmServeSession")
    def test_local_pipeline_batches_concretize_and_extract_globally_across_statements(
        self,
        mock_session_cls,
        _mock_covering_configs,
        _mock_coverage_stats,
        sample_spec_jsonl,
        tmp_path,
    ):
        class FakeSession:
            def __init__(self, outputs):
                self.outputs = list(outputs)
                self.batch_sizes: list[int] = []
                self.stage_names: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def generate_from_messages(self, message_batches, *, stage_name, temperature, max_tokens, n):
                self.batch_sizes.append(len(message_batches))
                self.stage_names.append(stage_name)
                next_outputs = self.outputs.pop(0)
                return [[text] for text in next_outputs]

            def metrics_snapshot(self):
                return {
                    "backend": "vllm_serve",
                    "model": "gs://bucket/shared",
                    "tensor_parallel_size": 4,
                    "max_model_len": 4096,
                    "tokenizer_load_seconds": 0.0,
                    "server_start_seconds": 0.0,
                    "session_enter_seconds": 0.0,
                    "totals": {},
                    "stages": {stage_name: {} for stage_name in self.stage_names},
                }

        stage1_helpful = (
            "<behavior_understanding>Being helpful means assisting users.</behavior_understanding>\n"
            "<scientific_motivation>Testing helpfulness is important.</scientific_motivation>\n"
            "<variation_axes>\n"
            '[{"axis": "request_complexity", "spectrum": ["simple", "complex"], "description": "Complexity"}]\n'
            "</variation_axes>"
        )
        stage1_honest = (
            "<behavior_understanding>Being honest means not fabricating claims.</behavior_understanding>\n"
            "<scientific_motivation>Testing honesty is important.</scientific_motivation>\n"
            "<variation_axes>\n"
            '[{"axis": "request_complexity", "spectrum": ["simple", "complex"], "description": "Complexity"}]\n'
            "</variation_axes>"
        )
        stage2_helpful = (
            "<scenario>A student asks for help with fractions.</scenario>\n"
            "<rubric>GOOD: Explain clearly. BAD: Refuse.</rubric>\n"
        )
        stage2_honest = (
            "<scenario>A user asks whether you know something uncertain.</scenario>\n"
            "<rubric>GOOD: Acknowledge uncertainty. BAD: Pretend certainty.</rubric>\n"
        )
        stage3_helpful = (
            "<system_prompt>You are a math tutor.</system_prompt>\n"
            "<user_message>Can you help me with fractions?</user_message>\n"
        )
        stage3_honest = (
            "<system_prompt>You are a careful assistant.</system_prompt>\n"
            "<user_message>Do you know for sure that this claim is true?</user_message>\n"
        )
        session = FakeSession(
            outputs=[
                [stage1_helpful, stage1_honest],
                [stage2_helpful, stage2_honest],
                [stage3_helpful, stage3_honest],
            ]
        )
        mock_session_cls.return_value = session

        shared_model = VLLMConfig(model="gs://bucket/shared", tensor_parallel_size=4, tpu_type="v5p-8")
        output_path = str(tmp_path / "prompt_local_global_batches")

        from marin.alignment.generate_prompts import generate_prompts_from_spec

        generate_prompts_from_spec(
            PromptGenConfig(
                spec_path=str(sample_spec_jsonl),
                output_path=output_path,
                ideation_model=shared_model,
                extract_model=shared_model,
                covering_strength=2,
                local_serve_batch_size=8,
            )
        )

        assert mock_session_cls.call_count == 1
        assert session.batch_sizes == [2, 2, 2]
        assert session.stage_names == ["understanding", "concretize", "extract"]
        prompts = load_sharded_jsonl_gz(output_path)
        assert {prompt["behavior_id"] for prompt in prompts} == {"be_helpful", "be_honest"}
        assert len(prompts) == 2

    @patch("marin.alignment.generate_prompts.compute_coverage_stats", return_value={"covered": 2})
    @patch(
        "marin.alignment.generate_prompts.generate_covering_configs",
        return_value=[
            {"request_complexity": "simple"},
            {"request_complexity": "complex"},
        ],
    )
    @patch("marin.alignment.generate_prompts.BatchedVllmServeSession")
    def test_local_pipeline_retries_missing_concretize_configs(
        self,
        mock_session_cls,
        _mock_covering_configs,
        _mock_coverage_stats,
        sample_spec_jsonl,
        tmp_path,
    ):
        class FakeSession:
            def __init__(self, outputs):
                self.outputs = list(outputs)
                self.batch_sizes: list[int] = []
                self.stage_names: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def generate_from_messages(self, message_batches, *, stage_name, temperature, max_tokens, n):
                self.batch_sizes.append(len(message_batches))
                self.stage_names.append(stage_name)
                next_outputs = self.outputs.pop(0)
                return [[text] for text in next_outputs]

            def metrics_snapshot(self):
                return {
                    "backend": "vllm_serve",
                    "model": "gs://bucket/shared",
                    "tensor_parallel_size": 4,
                    "max_model_len": 4096,
                    "tokenizer_load_seconds": 0.0,
                    "server_start_seconds": 0.0,
                    "session_enter_seconds": 0.0,
                    "totals": {},
                    "stages": {stage_name: {} for stage_name in self.stage_names},
                }

        session_instances: list[FakeSession] = []
        stage1_content = (
            "<behavior_understanding>Being helpful means assisting users.</behavior_understanding>\n"
            "<scientific_motivation>Testing helpfulness is important.</scientific_motivation>\n"
            "<variation_axes>\n"
            '[{"axis": "request_complexity", "spectrum": ["simple", "complex"], "description": "Complexity"}]\n'
            "</variation_axes>"
        )
        stage2_partial_content = (
            "<scenario>A student asks a simple question.</scenario>\n"
            "<rubric>GOOD: Answer directly. BAD: Refuse.</rubric>\n"
        )
        stage2_missing_content = ""
        stage2_retry_content = (
            "<scenario>A student asks a hard question.</scenario>\n"
            "<rubric>GOOD: Reason carefully. BAD: Hallucinate.</rubric>\n"
        )
        stage3_content_0 = (
            "<system_prompt>You are a math tutor.</system_prompt>\n"
            "<user_message>Can you help me with the easy problem?</user_message>\n"
        )
        stage3_content_1 = (
            "<system_prompt>You are a math tutor.</system_prompt>\n"
            "<user_message>Can you help me with the hard problem?</user_message>\n"
        )

        def session_factory(_config):
            session = FakeSession(
                outputs=[
                    [stage1_content],
                    [stage2_partial_content, stage2_missing_content],
                    [stage2_retry_content],
                    [stage3_content_0, stage3_content_1],
                ]
            )
            session_instances.append(session)
            return session

        mock_session_cls.side_effect = session_factory

        shared_model = VLLMConfig(model="gs://bucket/shared", tensor_parallel_size=4, tpu_type="v5p-8")
        output_path = str(tmp_path / "prompt_local_retry")

        from marin.alignment.generate_prompts import generate_prompts_from_spec

        generate_prompts_from_spec(
            PromptGenConfig(
                spec_path=str(sample_spec_jsonl),
                output_path=output_path,
                ideation_model=shared_model,
                extract_model=shared_model,
                covering_strength=2,
                local_serve_batch_size=4,
                statement_ids=["be_helpful"],
                concretize_max_attempts=2,
            )
        )

        assert mock_session_cls.call_count == 1
        assert session_instances[0].batch_sizes == [1, 2, 1, 2]
        prompts = load_sharded_jsonl_gz(output_path)
        assert len(prompts) == 2

        ideation_path = Path(output_path) / "artifacts" / "be_helpful" / "ideation.json"
        ideation = json.loads(ideation_path.read_text())
        assert [attempt["missing_config_ids"] for attempt in ideation["concretization_attempts"]] == [
            [],
            ["cfg_001"],
            [],
        ]
        assert ideation["variations"][1]["config_id"] == "cfg_001"
        assert ideation["variations"][1]["description"] == "A student asks a hard question."
        metrics_payload = json.loads((Path(output_path) / "artifacts" / "vllm_metrics.json").read_text())
        assert metrics_payload["logical_stage"] == "prompt_generation"

    @patch("marin.alignment.generate_prompts.compute_coverage_stats", return_value={"covered": 1})
    @patch(
        "marin.alignment.generate_prompts.generate_covering_configs",
        return_value=[{"request_complexity": "simple"}],
    )
    @patch("marin.alignment.generate_prompts.BatchedVllmServeSession")
    def test_local_pipeline_retries_missing_stage1_variation_axes(
        self,
        mock_session_cls,
        _mock_covering_configs,
        _mock_coverage_stats,
        sample_spec_jsonl,
        tmp_path,
    ):
        class FakeSession:
            def __init__(self, outputs):
                self.outputs = list(outputs)
                self.batch_sizes: list[int] = []
                self.stage_names: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def generate_from_messages(self, message_batches, *, stage_name, temperature, max_tokens, n):
                self.batch_sizes.append(len(message_batches))
                self.stage_names.append(stage_name)
                next_outputs = self.outputs.pop(0)
                return [[text] for text in next_outputs]

            def metrics_snapshot(self):
                return {
                    "backend": "vllm_serve",
                    "model": "gs://bucket/shared",
                    "tensor_parallel_size": 4,
                    "max_model_len": 4096,
                    "tokenizer_load_seconds": 0.0,
                    "server_start_seconds": 0.0,
                    "session_enter_seconds": 0.0,
                    "totals": {},
                    "stages": {stage_name: {} for stage_name in self.stage_names},
                }

        stage1_missing_axes = (
            "<behavior_understanding>Being helpful means assisting users.</behavior_understanding>\n"
            "<scientific_motivation>Testing helpfulness is important.</scientific_motivation>\n"
        )
        stage1_retry_content = (
            "<behavior_understanding>Being helpful means assisting users.</behavior_understanding>\n"
            "<scientific_motivation>Testing helpfulness is important.</scientific_motivation>\n"
            "<variation_axes>\n"
            '[{"axis": "request_complexity", "spectrum": ["simple", "complex"], "description": "Complexity"}]\n'
            "</variation_axes>"
        )
        stage2_content = (
            "<scenario>A student asks a simple question.</scenario>\n"
            "<rubric>GOOD: Answer directly. BAD: Refuse.</rubric>\n"
        )
        stage3_content = (
            "<system_prompt>You are a math tutor.</system_prompt>\n"
            "<user_message>Can you help me with the easy problem?</user_message>\n"
        )

        session = FakeSession(
            outputs=[
                [stage1_missing_axes],
                [stage1_retry_content],
                [stage2_content],
                [stage3_content],
            ]
        )
        mock_session_cls.return_value = session

        shared_model = VLLMConfig(model="gs://bucket/shared", tensor_parallel_size=4, tpu_type="v5p-8")
        output_path = str(tmp_path / "prompt_local_stage1_retry")

        from marin.alignment.generate_prompts import generate_prompts_from_spec

        generate_prompts_from_spec(
            PromptGenConfig(
                spec_path=str(sample_spec_jsonl),
                output_path=output_path,
                ideation_model=shared_model,
                extract_model=shared_model,
                covering_strength=2,
                local_serve_batch_size=4,
                statement_ids=["be_helpful"],
                understanding_max_attempts=2,
            )
        )

        assert mock_session_cls.call_count == 1
        assert session.batch_sizes == [1, 1, 1, 1]
        assert session.stage_names == ["understanding", "understanding", "concretize", "extract"]
        prompts = load_sharded_jsonl_gz(output_path)
        assert len(prompts) == 1
        assert prompts[0]["behavior_id"] == "be_helpful"

    @patch("marin.alignment.generate_prompts.compute_coverage_stats", return_value={"covered": 1})
    @patch(
        "marin.alignment.generate_prompts.generate_covering_configs",
        return_value=[{"request_complexity": "simple"}],
    )
    @patch("marin.alignment.generate_prompts.BatchedVllmServeSession")
    def test_local_pipeline_recovers_stage1_from_raw_attempts_and_only_reruns_pending_statements(
        self,
        mock_session_cls,
        _mock_covering_configs,
        _mock_coverage_stats,
        sample_spec_jsonl,
        tmp_path,
    ):
        class FakeSession:
            def __init__(self, outputs):
                self.outputs = list(outputs)
                self.batch_sizes: list[int] = []
                self.stage_names: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def generate_from_messages(self, message_batches, *, stage_name, temperature, max_tokens, n):
                self.batch_sizes.append(len(message_batches))
                self.stage_names.append(stage_name)
                next_outputs = self.outputs.pop(0)
                return [[text] for text in next_outputs]

            def metrics_snapshot(self):
                return {
                    "backend": "vllm_serve",
                    "model": "gs://bucket/shared",
                    "tensor_parallel_size": 4,
                    "max_model_len": 4096,
                    "tokenizer_load_seconds": 0.0,
                    "server_start_seconds": 0.0,
                    "session_enter_seconds": 0.0,
                    "totals": {},
                    "stages": {stage_name: {} for stage_name in self.stage_names},
                }

        stage1_helpful_content = (
            "<behavior_understanding>Being helpful means assisting users.</behavior_understanding>\n"
            "<scientific_motivation>Testing helpfulness is important.</scientific_motivation>\n"
            "<variation_axes>\n"
            '[{"axis": "request_complexity", "spectrum": ["simple", "complex"], "description": "Complexity"}]\n'
            "</variation_axes>"
        )
        stage1_honest_missing_axes = (
            "<behavior_understanding>Being honest means being truthful.</behavior_understanding>\n"
            "<scientific_motivation>Testing honesty is important.</scientific_motivation>\n"
        )
        stage1_honest_retry_content = (
            "<behavior_understanding>Being honest means being truthful.</behavior_understanding>\n"
            "<scientific_motivation>Testing honesty is important.</scientific_motivation>\n"
            "<variation_axes>\n"
            '[{"axis": "request_complexity", "spectrum": ["simple", "complex"], "description": "Complexity"}]\n'
            "</variation_axes>"
        )
        stage2_helpful_content = (
            "<scenario>A user asks for help with a simple task.</scenario>\n"
            "<rubric>GOOD: Helps clearly. BAD: Refuses.</rubric>\n"
        )
        stage2_honest_content = (
            "<scenario>A user asks for an uncertain fact.</scenario>\n"
            "<rubric>GOOD: Admits uncertainty. BAD: Hallucinates.</rubric>\n"
        )
        stage3_helpful_content = (
            "<system_prompt>You are helpful.</system_prompt>\n"
            "<user_message>Can you help with this task?</user_message>\n"
        )
        stage3_honest_content = (
            "<system_prompt>You are honest.</system_prompt>\n"
            "<user_message>What is the answer to this uncertain question?</user_message>\n"
        )

        session_instances: list[FakeSession] = []

        def session_factory(_config):
            outputs = (
                [[stage1_helpful_content, stage1_honest_missing_axes]]
                if not session_instances
                else [
                    [stage1_honest_retry_content],
                    [stage2_helpful_content, stage2_honest_content],
                    [stage3_helpful_content, stage3_honest_content],
                ]
            )
            session = FakeSession(outputs=outputs)
            session_instances.append(session)
            return session

        mock_session_cls.side_effect = session_factory

        shared_model = VLLMConfig(model="gs://bucket/shared", tensor_parallel_size=4, tpu_type="v5p-8")
        output_path = str(tmp_path / "prompt_local_stage1_attempt_resume")

        from marin.alignment.generate_prompts import generate_prompts_from_spec

        config = PromptGenConfig(
            spec_path=str(sample_spec_jsonl),
            output_path=output_path,
            ideation_model=shared_model,
            extract_model=shared_model,
            covering_strength=2,
            local_serve_batch_size=2,
            understanding_max_attempts=1,
        )

        with pytest.raises(RuntimeError, match="Stage 1 failed"):
            generate_prompts_from_spec(config)

        attempt_records = load_sharded_jsonl_gz(
            str(Path(output_path) / "artifacts" / "checkpoints" / "understanding_attempts")
        )
        assert len(attempt_records) == 2
        assert {record["statement_id"] for record in attempt_records} == {"be_helpful", "be_honest"}
        assert stage1_helpful_content in {record["raw_response"] for record in attempt_records}

        generate_prompts_from_spec(config)

        assert mock_session_cls.call_count == 2
        assert session_instances[0].stage_names == ["understanding"]
        assert session_instances[0].batch_sizes == [2]
        assert session_instances[1].stage_names == ["understanding", "concretize", "extract"]
        assert session_instances[1].batch_sizes == [1, 2, 2]

        prompts = load_sharded_jsonl_gz(output_path)
        assert len(prompts) == 2
        assert {prompt["behavior_id"] for prompt in prompts} == {"be_helpful", "be_honest"}

        attempt_records = load_sharded_jsonl_gz(
            str(Path(output_path) / "artifacts" / "checkpoints" / "understanding_attempts")
        )
        assert len(attempt_records) == 3

        stage1_checkpoint = json.loads(
            (Path(output_path) / "artifacts" / "be_helpful" / "understanding.json").read_text()
        )
        assert stage1_checkpoint["behavior_name"] == "be_helpful"

        stage_status = json.loads((Path(output_path) / "artifacts" / "checkpoints" / "stage_status.json").read_text())
        assert stage_status["understanding"]["complete"] is True
        assert stage_status["understanding"]["num_statements"] == 2

    @patch("marin.alignment.generate_prompts.compute_coverage_stats", return_value={"covered": 2})
    @patch(
        "marin.alignment.generate_prompts.generate_covering_configs",
        return_value=[
            {"request_complexity": "simple"},
            {"request_complexity": "complex"},
        ],
    )
    @patch("marin.alignment.generate_prompts.BatchedVllmServeSession")
    def test_local_pipeline_recovers_skipped_stage3_items_on_resume(
        self,
        mock_session_cls,
        _mock_covering_configs,
        _mock_coverage_stats,
        sample_spec_jsonl,
        tmp_path,
    ):
        class FakeSession:
            def __init__(self, outputs):
                self.outputs = list(outputs)
                self.batch_sizes: list[int] = []
                self.stage_names: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def generate_from_messages(self, message_batches, *, stage_name, temperature, max_tokens, n):
                self.batch_sizes.append(len(message_batches))
                self.stage_names.append(stage_name)
                next_outputs = self.outputs.pop(0)
                return [[text] for text in next_outputs]

            def metrics_snapshot(self):
                return {
                    "backend": "vllm_serve",
                    "model": "gs://bucket/shared",
                    "tensor_parallel_size": 4,
                    "max_model_len": 4096,
                    "tokenizer_load_seconds": 0.0,
                    "server_start_seconds": 0.0,
                    "session_enter_seconds": 0.0,
                    "totals": {},
                    "stages": {stage_name: {} for stage_name in self.stage_names},
                }

        stage1_content = (
            "<behavior_understanding>Being helpful means assisting users.</behavior_understanding>\n"
            "<scientific_motivation>Testing helpfulness is important.</scientific_motivation>\n"
            "<variation_axes>\n"
            '[{"axis": "request_complexity", "spectrum": ["simple", "complex"], "description": "Complexity"}]\n'
            "</variation_axes>"
        )
        stage2_content_0 = (
            "<scenario>A student asks a simple question.</scenario>\n"
            "<rubric>GOOD: Answer directly. BAD: Refuse.</rubric>\n"
        )
        stage2_content_1 = (
            "<scenario>A student asks a hard question.</scenario>\n"
            "<rubric>GOOD: Reason carefully. BAD: Hallucinate.</rubric>\n"
        )
        stage3_content_0 = (
            "<system_prompt>You are a math tutor.</system_prompt>\n"
            "<user_message>Can you help me with the easy problem?</user_message>\n"
        )
        stage3_missing_user = "<system_prompt>You are a math tutor.</system_prompt>\n"
        stage3_content_1 = (
            "<system_prompt>You are a math tutor.</system_prompt>\n"
            "<user_message>Can you help me with the hard problem?</user_message>\n"
        )

        session_instances: list[FakeSession] = []

        def session_factory(_config):
            outputs = (
                [
                    [stage1_content],
                    [stage2_content_0],
                    [stage2_content_1],
                    [stage3_content_0],
                    [stage3_missing_user],
                ]
                if not session_instances
                else [
                    [stage3_content_1],
                ]
            )
            session = FakeSession(outputs=outputs)
            session_instances.append(session)
            return session

        mock_session_cls.side_effect = session_factory

        shared_model = VLLMConfig(model="gs://bucket/shared", tensor_parallel_size=4, tpu_type="v5p-8")
        output_path = str(tmp_path / "prompt_local_stage3_resume")

        from marin.alignment.generate_prompts import generate_prompts_from_spec

        config = PromptGenConfig(
            spec_path=str(sample_spec_jsonl),
            output_path=output_path,
            ideation_model=shared_model,
            extract_model=shared_model,
            covering_strength=2,
            local_serve_batch_size=1,
            statement_ids=["be_helpful"],
            extract_max_attempts=1,
        )

        generate_prompts_from_spec(config)

        stage_status = json.loads((Path(output_path) / "artifacts" / "checkpoints" / "stage_status.json").read_text())
        assert stage_status["understanding"]["complete"] is True
        assert stage_status["concretize"]["complete"] is True
        assert stage_status["extract"]["complete"] is True

        extraction_shards = sorted((Path(output_path) / "artifacts" / "checkpoints" / "extractions").glob("*.jsonl.gz"))
        assert len(extraction_shards) == 1

        generate_prompts_from_spec(config)

        assert mock_session_cls.call_count == 2
        assert session_instances[0].stage_names == ["understanding", "concretize", "concretize", "extract", "extract"]
        assert session_instances[1].stage_names == ["extract"]
        assert session_instances[1].batch_sizes == [1]

        prompts = load_sharded_jsonl_gz(output_path)
        assert len(prompts) == 2
        assert {prompt["config_id"] for prompt in prompts} == {"cfg_000", "cfg_001"}

        stage_status = json.loads((Path(output_path) / "artifacts" / "checkpoints" / "stage_status.json").read_text())
        assert stage_status["extract"]["complete"] is True

    @patch("marin.alignment.generate_prompts.compute_coverage_stats", return_value={"covered": 2})
    @patch(
        "marin.alignment.generate_prompts.generate_covering_configs",
        return_value=[
            {"request_complexity": "simple"},
            {"request_complexity": "complex"},
        ],
    )
    @patch("marin.alignment.generate_prompts.BatchedVllmServeSession")
    def test_local_pipeline_retries_malformed_stage3_items_before_failing(
        self,
        mock_session_cls,
        _mock_covering_configs,
        _mock_coverage_stats,
        sample_spec_jsonl,
        tmp_path,
    ):
        class FakeSession:
            def __init__(self, outputs):
                self.outputs = list(outputs)
                self.batch_sizes: list[int] = []
                self.stage_names: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def generate_from_messages(self, message_batches, *, stage_name, temperature, max_tokens, n):
                self.batch_sizes.append(len(message_batches))
                self.stage_names.append(stage_name)
                next_outputs = self.outputs.pop(0)
                return [[text] for text in next_outputs]

            def metrics_snapshot(self):
                return {
                    "backend": "vllm_serve",
                    "model": "gs://bucket/shared",
                    "tensor_parallel_size": 4,
                    "max_model_len": 4096,
                    "tokenizer_load_seconds": 0.0,
                    "server_start_seconds": 0.0,
                    "session_enter_seconds": 0.0,
                    "totals": {},
                    "stages": {stage_name: {} for stage_name in self.stage_names},
                }

        stage1_content = (
            "<behavior_understanding>Being helpful means assisting users.</behavior_understanding>\n"
            "<scientific_motivation>Testing helpfulness is important.</scientific_motivation>\n"
            "<variation_axes>\n"
            '[{"axis": "request_complexity", "spectrum": ["simple", "complex"], "description": "Complexity"}]\n'
            "</variation_axes>"
        )
        stage2_content_0 = (
            "<scenario>A student asks a simple question.</scenario>\n"
            "<rubric>GOOD: Answer directly. BAD: Refuse.</rubric>\n"
        )
        stage2_content_1 = (
            "<scenario>A student asks a hard question.</scenario>\n"
            "<rubric>GOOD: Reason carefully. BAD: Hallucinate.</rubric>\n"
        )
        stage3_content_0 = (
            "<system_prompt>You are a math tutor.</system_prompt>\n"
            "<user_message>Can you help me with the easy problem?</user_message>\n"
        )
        stage3_missing_user = "<system_prompt>You are a math tutor.</system_prompt>\n"
        stage3_content_1 = (
            "<system_prompt>You are a math tutor.</system_prompt>\n"
            "<user_message>Can you help me with the hard problem?</user_message>\n"
        )

        session = FakeSession(
            outputs=[
                [stage1_content],
                [stage2_content_0],
                [stage2_content_1],
                [stage3_content_0],
                [stage3_missing_user],
                [stage3_content_1],
            ]
        )
        mock_session_cls.return_value = session

        shared_model = VLLMConfig(model="gs://bucket/shared", tensor_parallel_size=4, tpu_type="v5p-8")
        output_path = str(tmp_path / "prompt_local_stage3_retry")

        from marin.alignment.generate_prompts import generate_prompts_from_spec

        generate_prompts_from_spec(
            PromptGenConfig(
                spec_path=str(sample_spec_jsonl),
                output_path=output_path,
                ideation_model=shared_model,
                extract_model=shared_model,
                covering_strength=2,
                local_serve_batch_size=1,
                statement_ids=["be_helpful"],
                extract_max_attempts=2,
            )
        )

        assert mock_session_cls.call_count == 1
        assert session.stage_names == ["understanding", "concretize", "concretize", "extract", "extract", "extract"]
        prompts = load_sharded_jsonl_gz(output_path)
        assert len(prompts) == 2
        stage_status = json.loads((Path(output_path) / "artifacts" / "checkpoints" / "stage_status.json").read_text())
        assert stage_status["extract"]["complete"] is True

    @patch("marin.alignment.generate_prompts.compute_coverage_stats", return_value={"covered": 1})
    @patch(
        "marin.alignment.generate_prompts.generate_covering_configs",
        return_value=[{"request_complexity": "simple"}],
    )
    @patch("marin.alignment.generate_prompts.BatchedVllmServeSession")
    def test_local_pipeline_resumes_from_incremental_stage2_statement_checkpoints(
        self,
        mock_session_cls,
        _mock_covering_configs,
        _mock_coverage_stats,
        sample_spec_jsonl,
        tmp_path,
    ):
        class FakeSession:
            def __init__(self, outputs):
                self.outputs = list(outputs)
                self.batch_sizes: list[int] = []
                self.stage_names: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def generate_from_messages(self, message_batches, *, stage_name, temperature, max_tokens, n):
                self.batch_sizes.append(len(message_batches))
                self.stage_names.append(stage_name)
                next_outputs = self.outputs.pop(0)
                return [[text] for text in next_outputs]

            def metrics_snapshot(self):
                return {
                    "backend": "vllm_serve",
                    "model": "gs://bucket/shared",
                    "tensor_parallel_size": 4,
                    "max_model_len": 4096,
                    "tokenizer_load_seconds": 0.0,
                    "server_start_seconds": 0.0,
                    "session_enter_seconds": 0.0,
                    "totals": {},
                    "stages": {stage_name: {} for stage_name in self.stage_names},
                }

        stage1_helpful = (
            "<behavior_understanding>Being helpful means assisting users.</behavior_understanding>\n"
            "<scientific_motivation>Testing helpfulness is important.</scientific_motivation>\n"
            "<variation_axes>\n"
            '[{"axis": "request_complexity", "spectrum": ["simple", "complex"], "description": "Complexity"}]\n'
            "</variation_axes>"
        )
        stage1_honest = (
            "<behavior_understanding>Being honest means being truthful.</behavior_understanding>\n"
            "<scientific_motivation>Testing honesty is important.</scientific_motivation>\n"
            "<variation_axes>\n"
            '[{"axis": "request_complexity", "spectrum": ["simple", "complex"], "description": "Complexity"}]\n'
            "</variation_axes>"
        )
        stage2_helpful = (
            "<scenario>A user asks for help with a simple task.</scenario>\n"
            "<rubric>GOOD: Helps clearly. BAD: Refuses.</rubric>\n"
        )
        stage2_missing = ""
        stage2_honest_retry = (
            "<scenario>A user asks for an uncertain fact.</scenario>\n"
            "<rubric>GOOD: Admits uncertainty. BAD: Hallucinates.</rubric>\n"
        )
        stage3_helpful = (
            "<system_prompt>You are helpful.</system_prompt>\n"
            "<user_message>Can you help with this task?</user_message>\n"
        )
        stage3_honest = (
            "<system_prompt>You are honest.</system_prompt>\n"
            "<user_message>What is the answer to this uncertain question?</user_message>\n"
        )

        session_instances: list[FakeSession] = []

        def session_factory(_config):
            outputs = (
                [
                    [stage1_helpful, stage1_honest],
                    [stage2_helpful, stage2_missing],
                ]
                if not session_instances
                else [
                    [stage2_honest_retry],
                    [stage3_helpful, stage3_honest],
                ]
            )
            session = FakeSession(outputs=outputs)
            session_instances.append(session)
            return session

        mock_session_cls.side_effect = session_factory

        shared_model = VLLMConfig(model="gs://bucket/shared", tensor_parallel_size=4, tpu_type="v5p-8")
        output_path = str(tmp_path / "prompt_local_stage2_incremental_resume")

        from marin.alignment.generate_prompts import generate_prompts_from_spec

        config = PromptGenConfig(
            spec_path=str(sample_spec_jsonl),
            output_path=output_path,
            ideation_model=shared_model,
            extract_model=shared_model,
            covering_strength=2,
            local_serve_batch_size=2,
            concretize_max_attempts=1,
        )

        with pytest.raises(RuntimeError, match="Stage 2 failed"):
            generate_prompts_from_spec(config)

        checkpoint_dir = Path(output_path) / "artifacts" / "checkpoints" / "ideation_by_statement"
        helpful_checkpoint = checkpoint_dir / "be_helpful.json"
        honest_checkpoint = checkpoint_dir / "be_honest.json"
        assert helpful_checkpoint.exists()
        assert not honest_checkpoint.exists()

        helpful_record = json.loads(helpful_checkpoint.read_text())
        assert helpful_record["statement_id"] == "be_helpful"
        assert helpful_record["ideation"]["behavior_name"] == "be_helpful"
        assert len(helpful_record["ideation"]["concretization_attempts"]) == 1

        stage_status = json.loads((Path(output_path) / "artifacts" / "checkpoints" / "stage_status.json").read_text())
        assert stage_status["understanding"]["complete"] is True
        assert stage_status["concretize"]["complete"] is False

        generate_prompts_from_spec(config)

        assert mock_session_cls.call_count == 2
        assert session_instances[0].stage_names == ["understanding", "concretize"]
        assert session_instances[0].batch_sizes == [2, 2]
        assert session_instances[1].stage_names == ["concretize", "extract"]
        assert session_instances[1].batch_sizes == [1, 2]

        prompts = load_sharded_jsonl_gz(output_path)
        assert len(prompts) == 2
        assert {prompt["behavior_id"] for prompt in prompts} == {"be_helpful", "be_honest"}

        helpful_ideation = json.loads((Path(output_path) / "artifacts" / "be_helpful" / "ideation.json").read_text())
        assert len(helpful_ideation["concretization_attempts"]) == 1
        assert helpful_ideation["variations"][0]["description"] == "A user asks for help with a simple task."

    @patch("marin.alignment.generate_prompts.compute_coverage_stats", return_value={"covered": 1})
    @patch(
        "marin.alignment.generate_prompts.generate_covering_configs",
        return_value=[{"request_complexity": "simple"}],
    )
    @patch("marin.alignment.generate_prompts.BatchedVllmServeSession")
    def test_local_pipeline_discards_stale_incremental_stage2_statement_checkpoints(
        self,
        mock_session_cls,
        _mock_covering_configs,
        _mock_coverage_stats,
        sample_spec_jsonl,
        tmp_path,
    ):
        class FakeSession:
            def __init__(self, outputs):
                self.outputs = list(outputs)
                self.batch_sizes: list[int] = []
                self.stage_names: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def generate_from_messages(self, message_batches, *, stage_name, temperature, max_tokens, n):
                self.batch_sizes.append(len(message_batches))
                self.stage_names.append(stage_name)
                next_outputs = self.outputs.pop(0)
                return [[text] for text in next_outputs]

            def metrics_snapshot(self):
                return {
                    "backend": "vllm_serve",
                    "model": "gs://bucket/shared",
                    "tensor_parallel_size": 4,
                    "max_model_len": 4096,
                    "tokenizer_load_seconds": 0.0,
                    "server_start_seconds": 0.0,
                    "session_enter_seconds": 0.0,
                    "totals": {},
                    "stages": {stage_name: {} for stage_name in self.stage_names},
                }

        stage1_helpful = (
            "<behavior_understanding>Being helpful means assisting users.</behavior_understanding>\n"
            "<scientific_motivation>Testing helpfulness is important.</scientific_motivation>\n"
            "<variation_axes>\n"
            '[{"axis": "request_complexity", "spectrum": ["simple", "complex"], "description": "Complexity"}]\n'
            "</variation_axes>"
        )
        stage1_honest = (
            "<behavior_understanding>Being honest means being truthful.</behavior_understanding>\n"
            "<scientific_motivation>Testing honesty is important.</scientific_motivation>\n"
            "<variation_axes>\n"
            '[{"axis": "request_complexity", "spectrum": ["simple", "complex"], "description": "Complexity"}]\n'
            "</variation_axes>"
        )
        stage2_helpful = (
            "<scenario>A user asks for help with a simple task.</scenario>\n"
            "<rubric>GOOD: Helps clearly. BAD: Refuses.</rubric>\n"
        )
        stage2_missing = ""
        stage2_helpful_retry = (
            "<scenario>A user asks for help with a simple task.</scenario>\n"
            "<rubric>GOOD: Helps clearly. BAD: Refuses.</rubric>\n"
        )
        stage2_honest_retry = (
            "<scenario>A user asks for an uncertain fact.</scenario>\n"
            "<rubric>GOOD: Admits uncertainty. BAD: Hallucinates.</rubric>\n"
        )
        stage3_helpful = (
            "<system_prompt>You are helpful.</system_prompt>\n"
            "<user_message>Can you help with this task?</user_message>\n"
        )
        stage3_honest = (
            "<system_prompt>You are honest.</system_prompt>\n"
            "<user_message>What is the answer to this uncertain question?</user_message>\n"
        )

        session_instances: list[FakeSession] = []

        def session_factory(_config):
            outputs = (
                [
                    [stage1_helpful, stage1_honest],
                    [stage2_helpful, stage2_missing],
                ]
                if not session_instances
                else [
                    [stage2_helpful_retry, stage2_honest_retry],
                    [stage3_helpful, stage3_honest],
                ]
            )
            session = FakeSession(outputs=outputs)
            session_instances.append(session)
            return session

        mock_session_cls.side_effect = session_factory

        shared_model = VLLMConfig(model="gs://bucket/shared", tensor_parallel_size=4, tpu_type="v5p-8")
        output_path = str(tmp_path / "prompt_local_stage2_incremental_stale")

        from marin.alignment.generate_prompts import generate_prompts_from_spec

        config = PromptGenConfig(
            spec_path=str(sample_spec_jsonl),
            output_path=output_path,
            ideation_model=shared_model,
            extract_model=shared_model,
            covering_strength=2,
            local_serve_batch_size=2,
            concretize_max_attempts=1,
        )

        with pytest.raises(RuntimeError, match="Stage 2 failed"):
            generate_prompts_from_spec(config)

        helpful_checkpoint = (
            Path(output_path) / "artifacts" / "checkpoints" / "ideation_by_statement" / "be_helpful.json"
        )
        helpful_record = json.loads(helpful_checkpoint.read_text())
        helpful_record["fingerprint"]["plan_sha256"] = "stale"
        helpful_checkpoint.write_text(json.dumps(helpful_record, indent=2, sort_keys=True) + "\n")

        generate_prompts_from_spec(config)

        assert mock_session_cls.call_count == 2
        assert session_instances[1].stage_names == ["concretize", "extract"]
        assert session_instances[1].batch_sizes == [2, 2]

        prompts = load_sharded_jsonl_gz(output_path)
        assert len(prompts) == 2
        assert {prompt["behavior_id"] for prompt in prompts} == {"be_helpful", "be_honest"}


# ===========================================================================
# Tests: judge.py — end-to-end pair construction (mocked)
# ===========================================================================


class TestJudgeE2E:
    @patch("marin.alignment.judge.BatchedVllmServeSession")
    def test_local_batched_judge_writes_full_judgments_and_pairs(self, mock_session_cls, sample_spec_jsonl, tmp_path):
        chosen_record = {
            "prompt_id": "be_helpful/cfg_000",
            "behavior_id": "be_helpful",
            "system_prompt": "You are helpful.",
            "user_message": "Help me reset my password.",
            "rubric": "GOOD: clear steps. BAD: refusal or rudeness.",
            "responses": [
                {
                    "content": "Sure. Go to settings, choose reset password, and follow the email link.",
                    "index": 0,
                }
            ],
        }
        rejected_record = {
            "prompt_id": "be_helpful/cfg_000",
            "behavior_id": "be_helpful",
            "system_prompt": "You are helpful.",
            "user_message": "Help me reset my password.",
            "rubric": "GOOD: clear steps. BAD: refusal or rudeness.",
            "responses": [{"content": "I will not help you. Figure it out yourself.", "index": 0}],
        }
        chosen_path = tmp_path / "chosen"
        rejected_path = tmp_path / "rejected"
        judgments_path = tmp_path / "judgments"
        pairs_path = tmp_path / "pairs"
        write_sharded_jsonl_gz([chosen_record], str(chosen_path), shard_size=5000)
        write_sharded_jsonl_gz([rejected_record], str(rejected_path), shard_size=5000)

        mock_session = mock_session_cls.return_value.__enter__.return_value
        mock_session.generate_from_messages.return_value = [
            ['{"score": 9, "confidence": 0.9, "explanation": "Helpful", "highlights": []}'],
            ['{"score": 1, "confidence": 0.9, "explanation": "Rude", "highlights": []}'],
        ]
        mock_session.metrics_snapshot.return_value = {
            "backend": "vllm_serve",
            "model": "gs://bucket/model",
            "tensor_parallel_size": 4,
            "max_model_len": 4096,
            "tokenizer_load_seconds": 0.0,
            "server_start_seconds": 0.0,
            "session_enter_seconds": 0.0,
            "totals": {},
            "stages": {"judge": {}},
        }

        judge_responses(
            JudgeConfig(
                chosen_responses_path=str(chosen_path),
                rejected_responses_path=str(rejected_path),
                spec_path=str(sample_spec_jsonl),
                output_path=str(judgments_path),
                judge_model=VLLMConfig(model="gs://bucket/model", tensor_parallel_size=4, tpu_type="v5p-8"),
                batch_size=4,
            )
        )

        mock_session_cls.assert_called_once()
        mock_session.generate_from_messages.assert_called_once()
        call_kwargs = mock_session.generate_from_messages.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["n"] == 1

        written_judgments = load_sharded_jsonl_gz(str(judgments_path))
        assert len(written_judgments) == 1
        assert written_judgments[0]["status"] == "ok"
        assert written_judgments[0]["best_chosen"]["judgment"]["score"] == 9
        assert written_judgments[0]["worst_rejected"]["judgment"]["score"] == 1
        assert written_judgments[0]["gap"] == 8
        assert written_judgments[0]["chosen_candidates"][0]["judgment"]["raw_response"] is not None

        build_preference_pairs(
            PreferencePairFilterConfig(
                judgments_path=str(judgments_path),
                output_path=str(pairs_path),
                min_chosen_score=7.0,
                min_gap=2.0,
            )
        )

        written_pairs = load_sharded_jsonl_gz(str(pairs_path))
        assert len(written_pairs) == 1
        assert written_pairs[0]["chosen"][2]["content"].startswith("Sure.")
        assert written_pairs[0]["rejected"][2]["content"].startswith("I will not help")
        judgment_metrics = json.loads((judgments_path / "artifacts" / "vllm_metrics.json").read_text())
        assert judgment_metrics["logical_stage"] == "judgments"

        filter_decisions = load_sharded_jsonl_gz(str(pairs_path / "artifacts" / "filter_decisions"))
        assert len(filter_decisions) == 1
        assert filter_decisions[0]["passed"] is True
        assert filter_decisions[0]["reason"] == "passed"

    @patch("marin.alignment.judge.BatchedVllmServeSession")
    def test_local_batched_judge_logs_live_token_throughput(self, mock_session_cls, sample_spec_jsonl, tmp_path, caplog):
        chosen_records = [
            {
                "prompt_id": "be_helpful/cfg_000",
                "behavior_id": "be_helpful",
                "system_prompt": "You are helpful.",
                "user_message": "Help me reset my password.",
                "rubric": "GOOD: clear steps. BAD: refusal or rudeness.",
                "responses": [{"content": "Use the reset link in settings.", "index": 0}],
            },
            {
                "prompt_id": "be_honest/cfg_000",
                "behavior_id": "be_honest",
                "system_prompt": "You are honest.",
                "user_message": "Do you know the answer?",
                "rubric": "GOOD: admit uncertainty. BAD: bluff.",
                "responses": [{"content": "I am not fully sure; here is what I know.", "index": 0}],
            },
        ]
        rejected_records = [
            {
                "prompt_id": "be_helpful/cfg_000",
                "behavior_id": "be_helpful",
                "system_prompt": "You are helpful.",
                "user_message": "Help me reset my password.",
                "rubric": "GOOD: clear steps. BAD: refusal or rudeness.",
                "responses": [{"content": "No.", "index": 0}],
            },
            {
                "prompt_id": "be_honest/cfg_000",
                "behavior_id": "be_honest",
                "system_prompt": "You are honest.",
                "user_message": "Do you know the answer?",
                "rubric": "GOOD: admit uncertainty. BAD: bluff.",
                "responses": [{"content": "Definitely yes, even though I am guessing.", "index": 0}],
            },
        ]
        chosen_path = tmp_path / "chosen"
        rejected_path = tmp_path / "rejected"
        judgments_path = tmp_path / "judgments"
        write_sharded_jsonl_gz(chosen_records, str(chosen_path), shard_size=5000)
        write_sharded_jsonl_gz(rejected_records, str(rejected_path), shard_size=5000)

        mock_session = mock_session_cls.return_value.__enter__.return_value
        mock_session.generate_from_messages.side_effect = [
            [
                ['{"score": 9, "confidence": 0.9, "explanation": "Helpful", "highlights": []}'],
                ['{"score": 1, "confidence": 0.9, "explanation": "Rude", "highlights": []}'],
            ],
            [
                ['{"score": 8, "confidence": 0.8, "explanation": "Honest", "highlights": []}'],
                ['{"score": 2, "confidence": 0.8, "explanation": "Bluffing", "highlights": []}'],
            ],
        ]
        mock_session.metrics_snapshot.return_value = {
            "backend": "vllm_serve",
            "model": "gs://bucket/model",
            "tensor_parallel_size": 4,
            "max_model_len": 4096,
            "tokenizer_load_seconds": 0.0,
            "server_start_seconds": 0.0,
            "session_enter_seconds": 0.0,
            "totals": {},
            "stages": {
                "judge": {
                    "input_tokens_per_second": 1234.0,
                    "output_tokens_per_second": 321.0,
                }
            },
        }

        with caplog.at_level(logging.INFO):
            judge_responses(
                JudgeConfig(
                    chosen_responses_path=str(chosen_path),
                    rejected_responses_path=str(rejected_path),
                    spec_path=str(sample_spec_jsonl),
                    output_path=str(judgments_path),
                    judge_model=VLLMConfig(model="gs://bucket/model", tensor_parallel_size=4, tpu_type="v5p-8"),
                    batch_size=1,
                )
            )

        progress_messages = [record.message for record in caplog.records if "Judge progress:" in record.message]
        assert any("prompt 1.2k tok/s" in message for message in progress_messages)
        assert any("completion 321 tok/s" in message for message in progress_messages)

    def test_build_preference_pairs_preserves_filtered_decisions(self, tmp_path):
        judgments_path = tmp_path / "judgments"
        pairs_path = tmp_path / "pairs"
        judgment_records = [
            {
                "prompt_id": "pass",
                "behavior_id": "be_helpful",
                "system_prompt": "You are helpful.",
                "user_message": "Help me.",
                "rubric": "GOOD",
                "statement": {
                    "id": "be_helpful",
                    "text": "Helpful",
                    "type": "GUIDELINE",
                    "authority_level": "PLATFORM",
                    "section": "Core",
                    "subsection": "Help",
                },
                "status": "ok",
                "errors": [],
                "chosen_candidates": [],
                "rejected_candidates": [],
                "best_chosen": {
                    "response_index": 0,
                    "response_text": "Helpful answer",
                    "judgment": {
                        "score": 9,
                        "compliant": True,
                        "confidence": 0.9,
                        "explanation": "good",
                        "highlights": [],
                        "raw_response": "{}",
                    },
                },
                "worst_rejected": {
                    "response_index": 0,
                    "response_text": "Bad answer",
                    "judgment": {
                        "score": 1,
                        "compliant": False,
                        "confidence": 0.9,
                        "explanation": "bad",
                        "highlights": [],
                        "raw_response": "{}",
                    },
                },
                "gap": 8,
            },
            {
                "prompt_id": "low_gap",
                "behavior_id": "be_helpful",
                "system_prompt": "",
                "user_message": "Help me.",
                "rubric": "GOOD",
                "statement": {
                    "id": "be_helpful",
                    "text": "Helpful",
                    "type": "GUIDELINE",
                    "authority_level": "PLATFORM",
                    "section": "Core",
                    "subsection": "Help",
                },
                "status": "ok",
                "errors": [],
                "chosen_candidates": [],
                "rejected_candidates": [],
                "best_chosen": {
                    "response_index": 0,
                    "response_text": "Okay answer",
                    "judgment": {
                        "score": 7,
                        "compliant": True,
                        "confidence": 0.8,
                        "explanation": "ok",
                        "highlights": [],
                        "raw_response": "{}",
                    },
                },
                "worst_rejected": {
                    "response_index": 0,
                    "response_text": "Also okay answer",
                    "judgment": {
                        "score": 6,
                        "compliant": False,
                        "confidence": 0.8,
                        "explanation": "close",
                        "highlights": [],
                        "raw_response": "{}",
                    },
                },
                "gap": 1,
            },
            {
                "prompt_id": "missing_statement",
                "behavior_id": "missing",
                "system_prompt": "",
                "user_message": "Help me.",
                "rubric": "GOOD",
                "statement": None,
                "status": "missing_statement",
                "errors": ["No statement found"],
                "chosen_candidates": [],
                "rejected_candidates": [],
                "best_chosen": None,
                "worst_rejected": None,
                "gap": None,
            },
        ]
        write_sharded_jsonl_gz(judgment_records, str(judgments_path), shard_size=5000)

        build_preference_pairs(
            PreferencePairFilterConfig(
                judgments_path=str(judgments_path),
                output_path=str(pairs_path),
                min_chosen_score=7.0,
                min_gap=2.0,
            )
        )

        written_pairs = load_sharded_jsonl_gz(str(pairs_path))
        assert len(written_pairs) == 1
        assert written_pairs[0]["chosen"][2]["content"] == "Helpful answer"
        assert written_pairs[0]["rejected"][1]["content"] == "Help me."

        filter_decisions = load_sharded_jsonl_gz(str(pairs_path / "artifacts" / "filter_decisions"))
        reasons = {record["prompt_id"]: record["reason"] for record in filter_decisions}
        assert reasons["pass"] == "passed"
        assert reasons["low_gap"] == "low_gap"
        assert reasons["missing_statement"] == "status:missing_statement"

        filter_summary = json.loads((pairs_path / "artifacts" / "filter_summary.json").read_text())
        assert filter_summary["pair_count"] == 1
        assert filter_summary["decision_counts"]["passed"] == 1
        assert filter_summary["decision_counts"]["low_gap"] == 1
        assert filter_summary["decision_counts"]["status:missing_statement"] == 1

    @patch("marin.alignment.judge.llm_chat_single")
    def test_api_judge_writes_full_judgment_record(self, mock_llm, sample_spec_jsonl, tmp_path):
        call_count = {"n": 0}

        def mock_side_effect(**kwargs):
            call_count["n"] += 1
            score = 9 if call_count["n"] == 1 else 2
            return LLMResponse(
                content=json.dumps({"score": score, "confidence": 0.9, "explanation": "Test", "highlights": []}),
                model="gpt-4.1",
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
        chosen_path = tmp_path / "chosen"
        rejected_path = tmp_path / "rejected"
        judgments_path = tmp_path / "judgments"
        write_sharded_jsonl_gz([chosen_record], str(chosen_path), shard_size=5000)
        write_sharded_jsonl_gz([rejected_record], str(rejected_path), shard_size=5000)

        judge_responses(
            JudgeConfig(
                chosen_responses_path=str(chosen_path),
                rejected_responses_path=str(rejected_path),
                spec_path=str(sample_spec_jsonl),
                output_path=str(judgments_path),
                judge_model="gpt-4.1",
                workers=2,
                batch_size=4,
            )
        )

        written_judgments = load_sharded_jsonl_gz(str(judgments_path))
        assert len(written_judgments) == 1
        assert written_judgments[0]["best_chosen"]["response_text"] == "Of course, I'd be happy to help!"
        assert written_judgments[0]["worst_rejected"]["judgment"]["score"] == 2
