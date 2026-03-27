# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import fsspec
import pytest

from marin.rl.environments.inference_ctx.vllm import VLLMSamplingConfig, vLLMInferenceContextConfig
from marin.rl.rollout_worker import (
    _should_run_curriculum_eval,
    _should_run_micro_eval,
    _stage_vllm_metadata_locally,
    create_inference_context,
)


def test_stage_vllm_metadata_locally_copies_hf_metadata(tmp_path, monkeypatch):
    remote_dir = tmp_path / "remote-model"
    remote_dir.mkdir()
    (remote_dir / "config.json").write_text('{"architectures":["LlamaForCausalLM"],"model_type":"llama"}')
    (remote_dir / "tokenizer.json").write_text("{}")
    (remote_dir / "tokenizer_config.json").write_text("{}")

    monkeypatch.setattr(
        "marin.rl.rollout_worker.url_to_fs",
        lambda _path: (fsspec.filesystem("file"), str(remote_dir)),
    )
    monkeypatch.setattr("marin.rl.rollout_worker._VLLM_METADATA_CACHE_ROOT", str(tmp_path / "cache"))

    local_dir = Path(_stage_vllm_metadata_locally("gs://marin-us-central1/models/llama"))

    assert (local_dir / "config.json").exists()
    assert (local_dir / "tokenizer.json").exists()
    assert (local_dir / "tokenizer_config.json").exists()


def test_create_inference_context_uses_local_metadata_for_remote_inflight_vllm(monkeypatch):
    captured = {}

    class _FakeAsyncContext:
        def __init__(self, *, inference_config):
            captured["config"] = inference_config

    monkeypatch.setattr("marin.rl.rollout_worker._stage_vllm_metadata_locally", lambda _path: "/tmp/staged-model")
    monkeypatch.setattr("marin.rl.rollout_worker.AsyncvLLMInferenceContext", _FakeAsyncContext)

    ctx = create_inference_context(
        "vllm",
        vLLMInferenceContextConfig(
            model_name="gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f",
            canonical_model_name="meta-llama/Llama-3.1-8B-Instruct",
            max_model_len=2048,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.9,
            kv_cache_metrics=True,
            sampling_params=VLLMSamplingConfig(),
            load_format="runai_streamer",
        ),
        inflight_weight_updates=True,
    )

    assert isinstance(ctx, _FakeAsyncContext)
    assert captured["config"].model_name == "/tmp/staged-model"
    assert captured["config"].load_format == "dummy"
    assert captured["config"].kv_cache_metrics is True


@pytest.mark.parametrize(
    ("current_train_step", "last_eval_train_step", "eval_frequency", "worker_index", "expected"),
    [
        (-1, None, 1, 0, False),
        (0, None, 1, 0, True),
        (0, 0, 1, 0, False),
        (1, 0, 1, 0, True),
        (2, None, 4, 0, False),
        (4, None, 4, 0, True),
        (4, 4, 4, 0, False),
        (8, 4, 4, 0, True),
        (4, None, 4, 1, False),
        (8, 4, 4, 2, False),
    ],
)
def test_should_run_curriculum_eval(
    current_train_step: int,
    last_eval_train_step: int | None,
    eval_frequency: int,
    worker_index: int,
    expected: bool,
):
    assert (
        _should_run_curriculum_eval(
            current_train_step=current_train_step,
            last_eval_train_step=last_eval_train_step,
            eval_frequency=eval_frequency,
            worker_index=worker_index,
        )
        is expected
    )


def test_should_run_curriculum_eval_rejects_nonpositive_frequency():
    with pytest.raises(ValueError, match="eval_frequency must be positive"):
        _should_run_curriculum_eval(
            current_train_step=0,
            last_eval_train_step=None,
            eval_frequency=0,
            worker_index=0,
        )


@pytest.mark.parametrize(
    ("rollout_step", "micro_eval_frequency", "worker_index", "expected"),
    [
        (0, 10, 0, False),
        (1, 10, 0, False),
        (10, 10, 0, True),
        (20, 10, 0, True),
        (10, None, 0, False),
        (10, 10, 1, False),
    ],
)
def test_should_run_micro_eval(
    rollout_step: int,
    micro_eval_frequency: int | None,
    worker_index: int,
    expected: bool,
):
    assert (
        _should_run_micro_eval(
            rollout_step=rollout_step,
            micro_eval_frequency=micro_eval_frequency,
            worker_index=worker_index,
        )
        is expected
    )


def test_should_run_micro_eval_rejects_nonpositive_frequency():
    with pytest.raises(ValueError, match="micro_eval_frequency must be positive when enabled"):
        _should_run_micro_eval(
            rollout_step=10,
            micro_eval_frequency=0,
            worker_index=0,
        )
