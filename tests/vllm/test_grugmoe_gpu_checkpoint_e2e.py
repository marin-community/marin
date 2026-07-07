# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in checkpoint GrugMoE e2e for GPU vLLM and Levanter/JAX.

This is the CoreWeave/CUDA analogue of the TPU e2e. It validates the trained
checkpoint through the new vLLM PyTorch implementation and the JAX/Levanter
reference on GPUs.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.vllm import grugmoe_gpu_checkpoint_backend as backend
from tests.vllm.grugmoe_real_checkpoint_backend import E2EPaths, _expert_gate_up_tensors

BACKEND_PATH = Path(__file__).with_name("grugmoe_gpu_checkpoint_backend.py").resolve()
REPO_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def _repo_git_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
    ).strip()


def _resolve_run_id() -> str:
    run_id = os.environ.get(backend.RUN_ID_ENV)
    if run_id:
        return run_id
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _artifact_dir(run_id: str) -> str:
    return str(Path(backend.LOCAL_ARTIFACT_ROOT) / run_id / "artifact")


@pytest.fixture(scope="module")
def e2e_paths() -> E2EPaths:
    run_id = _resolve_run_id()
    output_dir = os.environ.get(backend.OUTPUT_DIR_ENV) or backend._join_path(backend.OUTPUT_ROOT, run_id)
    return E2EPaths(
        output_dir=output_dir,
        cache_dir=backend._join_path(backend.CACHE_ROOT, run_id),
        artifact_dir=_artifact_dir(run_id),
        export_result_path=backend._join_path(output_dir, "export-result.json"),
        vllm_result_path=backend._join_path(output_dir, "vllm-result-triton_attn.json"),
        levanter_result_path=backend._join_path(output_dir, "levanter-result.json"),
        summary_result_path=backend._join_path(output_dir, "result.json"),
    )


def _vllm_result_path(paths: E2EPaths, attention_backend: str) -> str:
    return backend._join_path(paths.output_dir, f"vllm-result-{attention_backend.lower()}.json")


def _vllm_result_paths(paths: E2EPaths) -> dict[str, str]:
    return {
        attention_backend: _vllm_result_path(paths, attention_backend)
        for attention_backend in backend.VLLM_ATTENTION_BACKENDS_UNDER_TEST
    }


def _run_subprocess_phase(
    phase: str,
    paths: E2EPaths,
    result_path: str,
    *,
    attention_backend: str = "",
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(BACKEND_PATH),
        "--phase",
        phase,
        "--checkpoint-path",
        backend.CHECKPOINT_PATH,
        "--tokenizer-path",
        backend.TOKENIZER_PATH,
        "--output-dir",
        paths.output_dir,
        "--artifact-dir",
        paths.artifact_dir,
        "--cache-dir",
        paths.cache_dir,
        "--result-path",
        result_path,
        "--levanter-result-path",
        paths.levanter_result_path,
    ]
    if phase == "vllm":
        command.extend(["--attention-backend", attention_backend])
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["CUDA_VISIBLE_DEVICES"] = backend.VISIBLE_CUDA_DEVICES
    env.setdefault("MARIN_GIT_SHA", _repo_git_sha())
    env["PYTHONPATH"] = os.pathsep.join(value for value in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if value)
    logger.info("grugmoe_gpu_checkpoint_e2e_command=%s", json.dumps(command))
    completed = subprocess.run(command, check=False, env=env, cwd=REPO_ROOT)
    if completed.returncode == 0:
        return backend._read_json(result_path)
    if backend._exists(result_path):
        result = backend._read_json(result_path)
        result["backend_returncode"] = completed.returncode
        return result
    completed.check_returncode()
    raise AssertionError("unreachable")


@pytest.fixture(scope="module")
def levanter_reference_setup_result(e2e_paths: E2EPaths) -> dict[str, Any]:
    return _run_subprocess_phase("export-and-levanter-reference", e2e_paths, e2e_paths.export_result_path)


@pytest.fixture(scope="module")
def vllm_results(e2e_paths: E2EPaths, levanter_reference_setup_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    del levanter_reference_setup_result
    return {
        attention_backend: _run_subprocess_phase(
            "vllm",
            e2e_paths,
            _vllm_result_path(e2e_paths, attention_backend),
            attention_backend=attention_backend,
        )
        for attention_backend in backend.VLLM_ATTENTION_BACKENDS_UNDER_TEST
    }


@pytest.fixture(scope="module")
def levanter_result(e2e_paths: E2EPaths, levanter_reference_setup_result: dict[str, Any]) -> dict[str, Any]:
    del levanter_reference_setup_result
    return backend._read_json(e2e_paths.levanter_result_path)


def _assert_levanter_golden_reference(
    levanter_reference_setup_result: dict[str, Any],
    levanter_result: dict[str, Any],
) -> None:
    assert levanter_reference_setup_result["jax_runtime"]["gpu_device_count"] >= backend.EXPECTED_GPU_COUNT
    assert levanter_reference_setup_result["jax_mesh"]["shape"]["expert"] == backend.LEVANTER_EXPERT_AXIS_SIZE

    assert levanter_result["prompt"] == backend.PROMPT
    assert levanter_result["max_new_tokens"] == backend.MAX_NEW_TOKENS
    assert levanter_result["expected_continuation"] == backend.EXPECTED_CONTINUATION
    assert levanter_result["completion"] == backend.EXPECTED_CONTINUATION

    expected_reference_completions = [backend.EXPECTED_CONTINUATION] * backend.LEVANTER_REFERENCE_REPEAT_COUNT
    reference_check_completions = [reference["completion"] for reference in levanter_result["reference_checks"]]
    assert len(set(levanter_result["reference_completions"])) == 1
    assert levanter_result["reference_completions"] == expected_reference_completions
    assert reference_check_completions == expected_reference_completions

    assert levanter_result["tokenization"]["prompt_token_count"] == len(
        levanter_result["tokenization"]["prompt_token_ids"]
    )
    assert levanter_result["tokenization"]["prompt_token_count"] > 0
    assert levanter_result["jax_runtime"]["gpu_device_count"] >= backend.EXPECTED_GPU_COUNT
    assert levanter_result["jax_mesh"]["device_count"] >= backend.EXPECTED_GPU_COUNT
    assert levanter_result["jax_mesh"]["shape"]["expert"] == backend.LEVANTER_EXPERT_AXIS_SIZE
    assert levanter_result["decode_batch_size"] == backend.EXPECTED_GPU_COUNT
    assert len(levanter_result["reference_checks"]) == backend.LEVANTER_REFERENCE_REPEAT_COUNT
    assert all(result["generated_token_ids"] for result in levanter_result["reference_checks"])
    assert all(result["serving_decode"]["skip_special_tokens"] is True for result in levanter_result["reference_checks"])


def _assert_vllm_result_matches_golden(
    attention_backend: str,
    vllm_result: dict[str, Any],
    levanter_result: dict[str, Any],
) -> None:
    assert "failure" not in vllm_result, vllm_result.get("failure")
    assert vllm_result["prompt"] == backend.PROMPT
    assert vllm_result["max_new_tokens"] == backend.MAX_NEW_TOKENS
    assert vllm_result["expected_continuation"] == backend.EXPECTED_CONTINUATION
    assert vllm_result["vllm_attention_backend"] == attention_backend
    assert vllm_result["torch_runtime"]["device_count"] >= backend.EXPECTED_GPU_COUNT
    assert all("H100" in device for device in vllm_result["torch_runtime"]["devices"])

    assert vllm_result["single_prompt_completion"] == levanter_result["completion"]
    assert vllm_result["single_prompt_completion"] == backend.EXPECTED_CONTINUATION
    assert vllm_result["single_prompt_output"]["prompt"] == backend.PROMPT
    assert vllm_result["single_prompt_output"]["completion"] == levanter_result["completion"]
    assert vllm_result["single_prompt_output"]["choice_summary"]["text"] == levanter_result["completion"]

    vllm_outputs = vllm_result["vllm_outputs"]
    vllm_output_completions = [output["completion"] for output in vllm_outputs]
    expected_batch_completions = [backend.EXPECTED_CONTINUATION] * backend.PROMPT_BATCH_SIZE
    assert len(vllm_outputs) == backend.PROMPT_BATCH_SIZE
    assert [output["prompt_index"] for output in vllm_outputs] == list(range(backend.PROMPT_BATCH_SIZE))
    assert [output["prompt"] for output in vllm_outputs] == [backend.PROMPT] * backend.PROMPT_BATCH_SIZE
    assert all(output["choice_summary"]["text"] == output["completion"] for output in vllm_outputs)
    assert vllm_result["completions"] == vllm_output_completions
    assert vllm_result["levanter_reference_completion"] == levanter_result["completion"]
    assert vllm_result["levanter_reference_outputs"] == levanter_result["reference_checks"]
    assert vllm_result["completions"] == [levanter_result["completion"]] * backend.PROMPT_BATCH_SIZE
    assert vllm_result["completions"] == expected_batch_completions
    assert vllm_result["completion"] == vllm_outputs[0]["completion"]

    assert vllm_result["requested_data_parallel_ranks"] == list(range(backend.VLLM_DATA_PARALLEL_SIZE))
    assert [batch["batch_size"] for batch in vllm_result["rank_request_batches"]] == [
        backend.PROMPTS_PER_VLLM_DATA_PARALLEL_RANK
    ] * backend.VLLM_DATA_PARALLEL_SIZE
    assert vllm_result["vllm_log_artifacts"]["copied"] is True
    assert "token_ids" in vllm_result["single_prompt_choice_summary"]
    assert all("token_ids" in summary for summary in vllm_result["main_choice_summaries"])


def _write_contract_summary(
    e2e_paths: E2EPaths,
    *,
    levanter_reference_setup_result: dict[str, Any],
    levanter_result: dict[str, Any],
    vllm_results: dict[str, dict[str, Any]],
) -> None:
    summary = {
        "passed": True,
        "checkpoint_path": backend.CHECKPOINT_PATH,
        "checkpoint_scope": backend.CHECKPOINT_SCOPE,
        "tokenizer_path": backend.TOKENIZER_PATH,
        "result_paths": {
            "export_and_levanter_reference": e2e_paths.export_result_path,
            "levanter": e2e_paths.levanter_result_path,
            "vllm": _vllm_result_paths(e2e_paths),
        },
        "artifact_dir": e2e_paths.artifact_dir,
        "prompt": backend.PROMPT,
        "expected_continuation": backend.EXPECTED_CONTINUATION,
        "max_new_tokens": backend.MAX_NEW_TOKENS,
        "golden_reference": {
            "source": "levanter",
            "completion": levanter_result["completion"],
            "reference_completions": levanter_result["reference_completions"],
            "reference_repeat_count": levanter_result["reference_repeat_count"],
        },
        "levanter": {
            "completion": levanter_result["completion"],
            "reference_repeat_count": levanter_result["reference_repeat_count"],
            "reference_completions": levanter_result["reference_completions"],
            "prompt_token_count": levanter_result["tokenization"]["prompt_token_count"],
            "prompt_token_ids": levanter_result["tokenization"]["prompt_token_ids"],
            "jax_gpu_device_count": levanter_result["jax_runtime"]["gpu_device_count"],
            "jax_mesh_shape": levanter_result["jax_mesh"]["shape"],
        },
        "vllm": {
            attention_backend: {
                "completion": result["completion"],
                "single_prompt_completion": result["single_prompt_completion"],
                "completions": result["completions"],
                "requested_data_parallel_ranks": result["requested_data_parallel_ranks"],
                "torch_device_count": result["torch_runtime"]["device_count"],
            }
            for attention_backend, result in sorted(vllm_results.items())
        },
        "attention_backends_tested": sorted(vllm_results),
        "reference_setup_elapsed_seconds": levanter_reference_setup_result.get("elapsed_seconds"),
    }
    backend._write_json(e2e_paths.summary_result_path, summary)
    logger.info("grugmoe_gpu_checkpoint_e2e_result=%s", json.dumps(summary, sort_keys=True))


def test_serving_completion_from_generated_ids_stops_at_eos() -> None:
    class FakeTokenizer:
        eos_token_id = 2

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is True
            vocab = {1: "A", 3: "B"}
            return "".join(vocab[token_id] for token_id in token_ids)

    completion, decode = backend._serving_completion_from_generated_ids(FakeTokenizer(), [1, 2, 3])

    assert completion == "A"
    assert decode == {
        "completion_token_ids": [1],
        "stop_token_id": 2,
        "stopped_on_eos": True,
        "skip_special_tokens": True,
    }


def test_expert_gate_up_tensors_accepts_current_split_layout() -> None:
    class SplitExpert:
        def __init__(self) -> None:
            self.w_gate = [[1, 2], [3, 4]]
            self.w_up = [[5, 6], [7, 8]]

    expert = SplitExpert()
    gate, up = _expert_gate_up_tensors(expert, intermediate_dim=2)

    assert gate is expert.w_gate
    assert up is expert.w_up


def test_expert_gate_up_tensors_accepts_legacy_fused_layout() -> None:
    class FusedExpert:
        def __init__(self) -> None:
            self.w_gate_up = np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])

    gate, up = _expert_gate_up_tensors(FusedExpert(), intermediate_dim=2)

    assert gate.tolist() == [[0, 1], [6, 7]]
    assert up.tolist() == [[2, 3, 4, 5], [8, 9, 10, 11]]


def test_vllm_server_logs_are_copied_to_output_prefix(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "stdout.log").write_text("stdout full log\n")
    (log_dir / "stderr.log").write_text("stderr full log\n")

    artifacts = backend._copy_vllm_server_logs(
        str(log_dir),
        str(tmp_path / "result-prefix"),
        attention_backend="TRITON_ATTN",
    )

    assert artifacts["copied"] is True
    assert artifacts["artifact_dir"] == str(tmp_path / "result-prefix" / "vllm-server-logs" / "triton_attn")
    assert {item["name"] for item in artifacts["files"]} == {"stdout.log", "stderr.log"}
    assert (
        tmp_path / "result-prefix" / "vllm-server-logs" / "triton_attn" / "stdout.log"
    ).read_text() == "stdout full log\n"
    assert (
        tmp_path / "result-prefix" / "vllm-server-logs" / "triton_attn" / "stderr.log"
    ).read_text() == "stderr full log\n"


def test_grugmoe_completion_request_sends_explicit_data_parallel_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeResponse:
        ok = True
        status_code = 200
        text = "{}"

        def json(self) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "text": backend.EXPECTED_CONTINUATION,
                        "finish_reason": "length",
                        "token_ids": [1, 2, 3, 4],
                    }
                ],
            }

        def raise_for_status(self) -> None:
            raise AssertionError("raise_for_status should not be called for successful fake response")

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(backend.requests, "post", fake_post)
    env = type("FakeEnv", (), {"server_url": "http://127.0.0.1:8000/v1", "model_id": "grugmoe"})()
    payload = backend._post_completion_request(
        env,
        prompts=[backend.PROMPT],
        data_parallel_rank=3,
        request_id="rank-3",
    )

    assert len(calls) == 1
    assert calls[0]["url"] == "http://127.0.0.1:8000/v1/completions"
    assert calls[0]["headers"] == {"X-data-parallel-rank": "3", "X-Request-Id": "rank-3"}
    assert calls[0]["json"]["prompt"] == [backend.PROMPT]
    assert calls[0]["json"]["add_special_tokens"] is backend.LEVANTER_PROMPT_ADD_SPECIAL_TOKENS
    assert calls[0]["json"]["return_token_ids"] is True
    assert payload["choices"][0]["text"] == backend.EXPECTED_CONTINUATION


def test_vllm_completion_batch_summary_returns_all_prompt_outputs() -> None:
    payloads: list[dict[str, Any]] = []
    rank_request_batches: list[dict[str, Any]] = []
    for data_parallel_rank in range(backend.VLLM_DATA_PARALLEL_SIZE):
        rank_start = data_parallel_rank * backend.PROMPTS_PER_VLLM_DATA_PARALLEL_RANK
        prompt_indices = list(range(rank_start, rank_start + backend.PROMPTS_PER_VLLM_DATA_PARALLEL_RANK))
        payloads.append(
            {
                "choices": [
                    {"text": f" output-{prompt_index}", "finish_reason": "length", "token_ids": [prompt_index]}
                    for prompt_index in prompt_indices
                ]
            }
        )
        rank_request_batches.append(
            {
                "data_parallel_rank": data_parallel_rank,
                "prompt_indices": prompt_indices,
                "batch_size": len(prompt_indices),
            }
        )
    batch = backend.VllmCompletionBatch(
        single_payload={
            "choices": [{"text": " single", "finish_reason": "length", "token_ids": [backend.PROMPT_BATCH_SIZE]}]
        },
        payloads=payloads,
        rank_request_batches=rank_request_batches,
    )

    summary = backend._summarize_vllm_completion_batch(batch)

    assert summary.single_completion == " single"
    assert summary.single_prompt_output["completion"] == " single"
    assert summary.completions == [f" output-{index}" for index in range(backend.PROMPT_BATCH_SIZE)]
    assert [output["prompt_index"] for output in summary.main_outputs] == list(range(backend.PROMPT_BATCH_SIZE))
    assert [output["data_parallel_rank"] for output in summary.main_outputs] == [
        index // backend.PROMPTS_PER_VLLM_DATA_PARALLEL_RANK for index in range(backend.PROMPT_BATCH_SIZE)
    ]


# Run this opt-in H100 contract with `-o addopts= --session-timeout=0 -m gpu_ci`;
# the marker timeout does not override the repo-wide pytest session timeout.
@pytest.mark.gpu_ci
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.data_integration
@pytest.mark.timeout(7200)
def test_grugmoe_gpu_checkpoint_contract(
    e2e_paths: E2EPaths,
    levanter_reference_setup_result: dict[str, Any],
    vllm_results: dict[str, dict[str, Any]],
    levanter_result: dict[str, Any],
) -> None:
    _assert_levanter_golden_reference(levanter_reference_setup_result, levanter_result)

    assert set(vllm_results) == set(backend.VLLM_ATTENTION_BACKENDS_UNDER_TEST)
    for attention_backend, vllm_result in vllm_results.items():
        _assert_vllm_result_matches_golden(attention_backend, vllm_result, levanter_result)

    _write_contract_summary(
        e2e_paths,
        levanter_reference_setup_result=levanter_reference_setup_result,
        levanter_result=levanter_result,
        vllm_results=vllm_results,
    )
