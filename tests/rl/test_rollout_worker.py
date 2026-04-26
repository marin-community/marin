# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import fsspec
import pytest

from marin.rl.environments.inference_ctx.staging import stage_vllm_metadata_locally
from marin.rl.environments.inference_ctx.vllm import (
    VLLMEngineConfig,
    VLLMFallbackSamplingConfig,
    vLLMInferenceContextConfig,
)
from marin.rl.run_state import RolloutTransferCounters
from marin.rl.rollout_worker import (
    RolloutTracker,
    RolloutTrackerConfig,
    RolloutTransferCounterSnapshot,
    RolloutWorker,
    _should_run_curriculum_eval,
    _should_run_micro_eval,
    create_inference_context,
)


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


def test_log_rollout_metrics_uses_wandb_default_step_and_logs_weight_train_steps():
    recorded_logs: list[tuple[dict[str, float | int], int | None]] = []

    class _FakeTracker:
        def log(self, metrics: dict[str, float | int], step=None):
            recorded_logs.append((metrics, step))

    worker = object.__new__(RolloutWorker)
    worker.config = SimpleNamespace(worker_index=1)
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
            None,
        )
    ]


def test_log_lesson_eval_uses_wandb_default_step_and_context_metrics():
    recorded_logs: list[tuple[dict[str, object], int | None]] = []

    class _FakeTracker:
        def log(self, metrics: dict[str, object], step=None):
            recorded_logs.append((metrics, step))

    worker = object.__new__(RolloutWorker)
    worker.config = SimpleNamespace(worker_index=0)
    worker._build_prompt_example_metrics = lambda lesson_id, batch, step, eval_type="eval": {
        f"inference.{eval_type}/{lesson_id}/sample_table": "table"
    }
    worker._current_train_step = 12
    worker._current_weight_step = -1
    worker.tracker = _FakeTracker()

    worker._log_lesson_eval(
        lesson_id="lesson-a",
        eval_type="eval",
        step=12,
        weight_step=-1,
        batch=SimpleNamespace(groups=[]),
        metrics={"inference.eval/lesson-a/avg_reward": 1.0},
    )

    assert recorded_logs == [
        (
            {
                "inference.eval/lesson-a/sample_table": "table",
                "inference.eval/lesson-a/avg_reward": 1.0,
                "inference.weight_step": -1,
                "inference.train_step": 12,
            },
            None,
        )
    ]


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
            engine=VLLMEngineConfig(
                model_name="gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f",
                canonical_model_name="meta-llama/Llama-3.1-8B-Instruct",
                max_model_len=2048,
                tensor_parallel_size=4,
                gpu_memory_utilization=0.9,
                kv_cache_metrics=True,
                load_format="runai_streamer",
            ),
            fallback_sampling=VLLMFallbackSamplingConfig(),
        ),
        inflight_weight_updates=True,
    )

    assert isinstance(ctx, _FakeAsyncContext)
    assert captured["config"].engine.model_name == "/tmp/staged-model"
    assert captured["config"].engine.load_format == "dummy"
    assert captured["config"].engine.kv_cache_metrics is True


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
