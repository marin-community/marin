# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import threading
import time
from pathlib import Path

import fsspec
import pytest

from marin.rl.environments.inference_ctx import PackedvLLMInferenceContextConfig
from marin.rl.environments.inference_ctx.staging import stage_vllm_metadata_locally
from marin.rl.environments.inference_ctx.vllm import VLLMSamplingConfig, vLLMInferenceContextConfig
from marin.rl.rollout_worker import (
    CompletedLessonEval,
    RolloutWorker,
    ScheduledEvalJob,
    _should_run_curriculum_eval,
    _should_run_micro_eval,
    _coalesce_eval_job,
    create_inference_context,
)
from marin.rl.weight_transfer import WeightTransferConfig


def test_stage_vllm_metadata_locally_copies_hf_metadata(tmp_path, monkeypatch):
    remote_dir = tmp_path / "remote-model"
    remote_dir.mkdir()
    (remote_dir / "config.json").write_text('{"architectures":["LlamaForCausalLM"],"model_type":"llama"}')
    (remote_dir / "tokenizer.json").write_text("{}")
    (remote_dir / "tokenizer_config.json").write_text("{}")

    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.staging.url_to_fs",
        lambda _path: (fsspec.filesystem("file"), str(remote_dir)),
    )
    monkeypatch.setattr("marin.rl.environments.inference_ctx.staging._VLLM_METADATA_CACHE_ROOT", str(tmp_path / "cache"))

    local_dir = Path(stage_vllm_metadata_locally("gs://marin-us-central1/models/llama"))

    assert (local_dir / "config.json").exists()
    assert (local_dir / "tokenizer.json").exists()
    assert (local_dir / "tokenizer_config.json").exists()


def test_create_inference_context_uses_local_metadata_for_remote_inflight_vllm(monkeypatch):
    captured = {}

    class _FakeAsyncContext:
        def __init__(self, *, inference_config):
            captured["config"] = inference_config

    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.staging.stage_vllm_metadata_locally",
        lambda _path: "/tmp/staged-model",
    )
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
        weight_transfer_config=WeightTransferConfig(),
        coordinator_handle=None,
    )

    assert isinstance(ctx, _FakeAsyncContext)
    assert captured["config"].model_name == "/tmp/staged-model"
    assert captured["config"].load_format == "dummy"
    assert captured["config"].kv_cache_metrics is True


def test_create_inference_context_uses_packed_vllm_context(monkeypatch):
    captured = {}

    class _FakePackedContext:
        def __init__(self, *, inference_config, inflight_weight_updates, weight_transfer_config, coordinator_handle):
            captured["config"] = inference_config
            captured["inflight_weight_updates"] = inflight_weight_updates
            captured["weight_transfer_config"] = weight_transfer_config
            captured["coordinator_handle"] = coordinator_handle

    monkeypatch.setattr("marin.rl.rollout_worker.PackedvLLMInferenceContext", _FakePackedContext)

    packed_config = PackedvLLMInferenceContextConfig(
        model_name="test-model",
        canonical_model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_model_len=2048,
        tensor_parallel_size_per_replica=2,
        gpu_memory_utilization=0.9,
        replica_chip_groups=("0,1", "2,3"),
        sampling_params=VLLMSamplingConfig(),
    )
    weight_transfer_config = WeightTransferConfig()

    ctx = create_inference_context(
        "vllm",
        packed_config,
        inflight_weight_updates=True,
        weight_transfer_config=weight_transfer_config,
        coordinator_handle="coord-handle",
    )

    assert isinstance(ctx, _FakePackedContext)
    assert captured["config"] == packed_config
    assert captured["inflight_weight_updates"] is True
    assert captured["weight_transfer_config"] == weight_transfer_config
    assert captured["coordinator_handle"] == "coord-handle"


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


def test_coalesce_eval_job_prefers_full_eval_and_newer_steps():
    existing = ScheduledEvalJob(eval_type="micro_eval", lesson_id="lesson-a", rng=1, step=10, tracker_step=10)
    replacement = ScheduledEvalJob(eval_type="eval", rng=2, step=12, tracker_step=12)

    assert _coalesce_eval_job(existing, replacement) == replacement


def test_async_eval_job_submission_coalesces_pending_work():
    worker = object.__new__(RolloutWorker)
    worker._eval_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    worker._eval_lock = threading.Lock()
    worker._active_eval_future = None
    worker._pending_eval_job = None
    worker._running = True

    started = threading.Event()
    release_first_job = threading.Event()
    events: list[tuple[str, str, int]] = []

    def fake_run_eval_job(job: ScheduledEvalJob) -> list[CompletedLessonEval]:
        events.append(("start", job.eval_type, job.step))
        if job.step == 1:
            started.set()
            assert release_first_job.wait(timeout=5)
        return []

    def fake_consume_eval_results(results: list[CompletedLessonEval]) -> None:
        events.append(("consume", "results", len(results)))

    worker._run_eval_job = fake_run_eval_job
    worker._consume_eval_results = fake_consume_eval_results

    first_job = ScheduledEvalJob(eval_type="micro_eval", lesson_id="lesson-a", rng=1, step=1, tracker_step=1)
    second_job = ScheduledEvalJob(eval_type="micro_eval", lesson_id="lesson-b", rng=2, step=2, tracker_step=2)
    third_job = ScheduledEvalJob(eval_type="eval", rng=3, step=3, tracker_step=3)

    try:
        worker._enqueue_eval_job(first_job)
        assert started.wait(timeout=5)

        worker._enqueue_eval_job(second_job)
        worker._enqueue_eval_job(third_job)
        assert worker._pending_eval_job == third_job

        release_first_job.set()
        deadline = time.time() + 5
        while time.time() < deadline:
            worker._poll_eval_jobs()
            with worker._eval_lock:
                active_done = worker._active_eval_future is None or worker._active_eval_future.done()
                pending_empty = worker._pending_eval_job is None
            if active_done and pending_empty:
                worker._poll_eval_jobs()
                break
            time.sleep(0.01)
        worker._poll_eval_jobs()
    finally:
        worker._running = False
        worker._eval_executor.shutdown(wait=True)

    assert ("start", "micro_eval", 1) in events
    assert ("start", "eval", 3) in events
    assert ("start", "micro_eval", 2) not in events
