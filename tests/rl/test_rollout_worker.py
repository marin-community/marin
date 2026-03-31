# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import fsspec
import pytest

from marin.rl.environments.inference_ctx import PackedvLLMInferenceContextConfig
from marin.rl.environments.inference_ctx.staging import stage_vllm_metadata_locally
from marin.rl.environments.inference_ctx.vllm import VLLMSamplingConfig, vLLMInferenceContextConfig
from marin.rl.run_state import RolloutTransferCounters
from marin.rl.rollout_worker import (
    CompletedLessonEval,
    RolloutBatchStats,
    RolloutTracker,
    RolloutTrackerConfig,
    RolloutTransferCounterSnapshot,
    RolloutWorker,
    ScheduledEvalJob,
    _should_run_curriculum_eval,
    _should_run_micro_eval,
    _coalesce_eval_job,
    create_inference_context,
)
from marin.rl.weight_transfer import WeightTransferConfig


def test_rollout_tracker_uses_explicit_name_when_provided(monkeypatch):
    captured = {}

    class _FakeRun:
        def log(self, _metrics, step=None):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(
        "marin.rl.rollout_worker.wandb.init",
        lambda **kwargs: captured.update(kwargs) or _FakeRun(),
    )

    RolloutTracker(
        RolloutTrackerConfig(project="marin_iris_rl_debug", name="iris-rl-e4ms2-500-rollout-0"),
        run_id="iris-rl-e4ms2-500-rollout-0",
    )

    assert captured["name"] == "iris-rl-e4ms2-500-rollout-0"
    assert captured["id"] == "iris-rl-e4ms2-500-rollout-0"
    assert captured["resume"] == "allow"


def test_resume_safe_transfer_metrics_logs_attempt_and_cumulative_values_after_counter_reset():
    recorded_calls: list[tuple[int, int, int, int]] = []

    class _FakeRemoteResult:
        def result(self) -> RolloutTransferCounters:
            return RolloutTransferCounters(total_polls=97, successful_receives=10, failed_receives=1)

    class _FakeRemoteMethod:
        def remote(
            self,
            worker_index: int,
            total_polls_delta: int,
            successful_receives_delta: int,
            failed_receives_delta: int,
        ) -> _FakeRemoteResult:
            recorded_calls.append((worker_index, total_polls_delta, successful_receives_delta, failed_receives_delta))
            return _FakeRemoteResult()

    class _FakeTransferClient:
        def get_metrics(self) -> dict[str, float | int]:
            return {
                "total_polls": 5,
                "successful_receives": 5,
                "failed_receives": 0,
                "total_receive_bytes": 4096,
                "receive_bytes": 1024,
                "param_count": 3,
                "largest_param_bytes": 512,
                "fetch_time": 1.25,
                "decode_time": 0.75,
                "poll_time": 0.5,
                "fetch_mib_per_second": 8.0,
                "decode_mib_per_second": 4.0,
            }

    worker = object.__new__(RolloutWorker)
    worker.config = SimpleNamespace(worker_index=1)
    worker._transfer_client = _FakeTransferClient()
    worker._runtime = SimpleNamespace(run_state=SimpleNamespace(add_rollout_transfer_counters=_FakeRemoteMethod()))
    worker._last_transfer_counters = RolloutTransferCounterSnapshot(
        total_polls=92,
        successful_receives=5,
        failed_receives=1,
    )

    metrics = worker._resume_safe_transfer_metrics()

    assert recorded_calls == [(1, 5, 0, 0)]
    assert metrics == {
        "attempt_total_polls": 5,
        "attempt_successful_receives": 5,
        "attempt_failed_receives": 0,
        "total_polls": 97,
        "successful_receives": 10,
        "failed_receives": 1,
        "total_receive_bytes": 4096,
        "receive_bytes": 1024,
        "param_count": 3,
        "largest_param_bytes": 512,
        "fetch_time": 1.25,
        "decode_time": 0.75,
        "poll_time": 0.5,
        "fetch_mib_per_second": 8.0,
        "decode_mib_per_second": 4.0,
    }


def test_log_rollout_metrics_uses_shared_tracker_step_and_logs_weight_train_steps():
    recorded_worker_indices: list[int] = []
    recorded_logs: list[tuple[dict[str, float | int], int | None]] = []

    class _FakeRemoteResult:
        def result(self) -> int:
            return 42

    class _FakeRemoteMethod:
        def remote(self, worker_index: int) -> _FakeRemoteResult:
            recorded_worker_indices.append(worker_index)
            return _FakeRemoteResult()

    class _FakeTracker:
        def log(self, metrics: dict[str, float | int], step=None):
            recorded_logs.append((metrics, step))

    worker = object.__new__(RolloutWorker)
    worker.config = SimpleNamespace(worker_index=1)
    worker._runtime = SimpleNamespace(run_state=SimpleNamespace(next_rollout_tracker_step=_FakeRemoteMethod()))
    worker._resume_safe_transfer_metrics = lambda: {"successful_receives": 10}
    worker._policy_ctx = SimpleNamespace(get_metrics=lambda: {"cache_hits": 3})
    worker._rollout_writer = SimpleNamespace(get_metrics=lambda: {"queued_batches": 2})
    worker._current_weight_step = -1
    worker._current_train_step = 248
    worker.tracker = _FakeTracker()

    worker._log_rollout_metrics(
        rollout_metrics={"rollout/math/pass_at_1": 0.5},
        env_metrics={"episodes": 7},
        throughput_metrics={"inference.throughput/batch_time_seconds": 2.0},
        rollout_step=30,
    )

    assert recorded_worker_indices == [1]
    assert recorded_logs == [
        (
            {
                "inference.rollout/math/pass_at_1": 0.5,
                "inference.successful_receives": 10,
                "inference.cache_hits": 3,
                "inference.env.episodes": 7,
                "inference.queued_batches": 2,
                "inference.throughput/batch_time_seconds": 2.0,
                "inference.weight_step": -1,
                "inference.train_step": 248,
            },
            42,
        )
    ]


def test_consume_lesson_eval_logs_with_shared_tracker_step_and_context_metrics():
    recorded_worker_indices: list[int] = []
    recorded_logs: list[tuple[dict[str, object], int | None]] = []
    recorded_curriculum_updates: list[tuple[list[object], str, int]] = []

    class _FakeRemoteResult:
        def result(self):
            return 17

    class _FakeTrackerStepMethod:
        def remote(self, worker_index: int) -> _FakeRemoteResult:
            recorded_worker_indices.append(worker_index)
            return _FakeRemoteResult()

    class _FakeCurriculumResult:
        def result(self):
            return None

    class _FakeCurriculumMethod:
        def remote(self, rollout_stats: list[object], mode: str, current_step: int) -> _FakeCurriculumResult:
            recorded_curriculum_updates.append((rollout_stats, mode, current_step))
            return _FakeCurriculumResult()

    class _FakeTracker:
        def log(self, metrics: dict[str, object], step=None):
            recorded_logs.append((metrics, step))

    worker = object.__new__(RolloutWorker)
    worker.config = SimpleNamespace(worker_index=0)
    worker._runtime = SimpleNamespace(run_state=SimpleNamespace(next_rollout_tracker_step=_FakeTrackerStepMethod()))
    worker._curriculum_actor = SimpleNamespace(update_lesson_stats=_FakeCurriculumMethod())
    worker._build_prompt_example_metrics = lambda lesson_id, batch, step, eval_type="eval": {
        f"inference.{eval_type}/{lesson_id}/sample_table": "table"
    }
    worker.tracker = _FakeTracker()

    rollout_stats = [SimpleNamespace()]
    result = CompletedLessonEval(
        lesson_id="lesson-a",
        eval_type="eval",
        step=12,
        weight_step=-1,
        batch=SimpleNamespace(groups=[]),
        stats=RolloutBatchStats(
            total_count=1,
            success_count=1,
            rollout_stats=rollout_stats,
            avg_reward=1.0,
            pass_at_one=1.0,
            pass_at_k=1.0,
            avg_at_k=1.0,
        ),
        metrics={"inference.eval/lesson-a/avg_reward": 1.0},
    )

    worker._consume_lesson_eval(result)

    assert recorded_worker_indices == [0]
    assert recorded_logs == [
        (
            {
                "inference.eval/lesson-a/sample_table": "table",
                "inference.eval/lesson-a/avg_reward": 1.0,
                "inference.weight_step": -1,
                "inference.train_step": 12,
            },
            17,
        )
    ]
    assert recorded_curriculum_updates == [(rollout_stats, "eval", 12)]


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
    existing = ScheduledEvalJob(eval_type="micro_eval", lesson_id="lesson-a", rng=1, step=10, weight_step=10)
    replacement = ScheduledEvalJob(eval_type="eval", rng=2, step=12, weight_step=12)

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

    first_job = ScheduledEvalJob(eval_type="micro_eval", lesson_id="lesson-a", rng=1, step=1, weight_step=1)
    second_job = ScheduledEvalJob(eval_type="micro_eval", lesson_id="lesson-b", rng=2, step=2, weight_step=2)
    third_job = ScheduledEvalJob(eval_type="eval", rng=3, step=3, weight_step=3)

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
