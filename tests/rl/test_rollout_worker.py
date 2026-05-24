# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import fsspec
import numpy as np
import pytest
from marin.rl.curriculum import CurriculumConfig, EvalSamplingParams, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig, FiniteDatasetEnv
from marin.rl.environments.inference_ctx.staging import stage_vllm_metadata_locally
from marin.rl.environments.inference_ctx.vllm import VLLMSamplingConfig, vLLMInferenceContextConfig
from marin.rl.rollout_schedule import rollout_assignment
from marin.rl.rollout_worker import (
    RolloutTracker,
    RolloutTrackerConfig,
    RolloutTransferCounterSnapshot,
    RolloutWorker,
    _rollout_batch_throughput_and_length_metrics,
    _sample_table_metric_key,
    _should_run_curriculum_eval,
    _should_run_micro_eval,
    create_inference_context,
)
from marin.rl.run_state import RolloutTransferCounters
from marin.rl.types import Rollout, RolloutBatch, RolloutGroup, RolloutMetadata
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage


class _FakeEvalFiniteEnv(FiniteDatasetEnv):
    def __init__(self, eval_len: int = 3):
        self.train_examples = [f"train-{idx}" for idx in range(eval_len)]
        self.eval_examples = [f"eval-{idx}" for idx in range(eval_len)]

    @property
    def env_name(self) -> str:
        return "fake"

    def train_len(self) -> int:
        return len(self.train_examples)

    def eval_len(self) -> int:
        return len(self.eval_examples)

    def train_examples_by_indices(self, indices):
        return [self.train_examples[idx] for idx in indices]

    def eval_examples_by_indices(self, indices):
        return [self.eval_examples[idx] for idx in indices]

    def inference_prompt_for_example(self, example):
        return example

    def rollout_prompt_for_example(self, example) -> str:
        return example

    def example_id(self, example) -> str:
        return example.replace("-", "_")

    def score_choice(self, example, response_text: str, finish_reason: str, tokenizer):
        return 1.0, 1.0, 1.0


class _FakeEvalInferenceContext:
    def __init__(self):
        self.calls = []
        self.tokenizer = object()

    def batch_completions(self, prompts, temperature, n, max_tokens=None, top_k=None, stop=None, system_prompt=None):
        self.calls.append(
            {
                "prompts": prompts,
                "temperature": temperature,
                "n": n,
                "max_tokens": max_tokens,
                "top_k": top_k,
                "stop": stop,
            }
        )
        return [_chat_completion(n) for _ in prompts]

    def create_rollout_from_choice(
        self,
        prompt,
        choice,
        env_name,
        env_example_id,
        reward,
        temperature,
        top_k=None,
        system_prompt=None,
        correctness_reward=None,
    ):
        return Rollout(
            env_name=env_name,
            env_example_id=env_example_id,
            prompt_tokens=np.array([1], dtype=np.int32),
            response_tokens=np.array([2], dtype=np.int32),
            response_logprobs=np.array([-0.1], dtype=np.float32),
            token_rewards=np.array([reward], dtype=np.float32),
            episode_reward=reward,
            temperature=temperature,
            top_k=top_k,
            is_truncated=False,
            correctness_reward=correctness_reward,
        )


class _FakeRemoteResult:
    def __init__(self, value=None):
        self.value = value

    def result(self, *args, **kwargs):
        return self.value


class _FakeUpdateLessonStats:
    def __init__(self):
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _FakeRemoteResult()


class _FakeRemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return _FakeRemoteResult(self.fn(*args, **kwargs))


def _chat_completion(n: int) -> ChatCompletion:
    return ChatCompletion(
        id="completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=idx,
                message=ChatCompletionMessage(role="assistant", content="ok"),
            )
            for idx in range(n)
        ],
        created=0,
        model="test",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=n, prompt_tokens=1, total_tokens=n + 1),
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
    worker._runtime = SimpleNamespace(
        run_state=SimpleNamespace(
            get_rollout_schedule_stats=_FakeRemoteMethod(
                lambda: {
                    "active_cursors": 2,
                    "pending_assignments": 1,
                    "reserved_assignments": 7,
                    "reused_pending_assignments": 3,
                    "committed_assignments": 6,
                    "ledger_recovered_assignments": 4,
                }
            )
        )
    )
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
                "inference.schedule/active_cursors": 2,
                "inference.schedule/pending_assignments": 1,
                "inference.schedule/reserved_assignments": 7,
                "inference.schedule/reused_pending_assignments": 3,
                "inference.schedule/committed_assignments": 6,
                "inference.schedule/ledger_recovered_assignments": 4,
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


def test_rollout_batch_throughput_metrics_include_length_cap_saturation():
    def _rollout(response_len: int, is_truncated: bool) -> Rollout:
        response_tokens = np.arange(response_len, dtype=np.int32)
        return Rollout(
            env_name="fake",
            env_example_id=f"example_{response_len}",
            prompt_tokens=np.array([1], dtype=np.int32),
            response_tokens=response_tokens,
            response_logprobs=np.zeros(response_len, dtype=np.float32),
            token_rewards=np.zeros(response_len, dtype=np.float32),
            episode_reward=0.0,
            temperature=1.0,
            top_k=5,
            is_truncated=is_truncated,
        )

    batch = RolloutBatch(
        groups=[
            RolloutGroup(
                rollouts=[
                    _rollout(94, False),
                    _rollout(95, False),
                    _rollout(100, True),
                ]
            )
        ],
        metadata=RolloutMetadata(),
    )

    metrics = _rollout_batch_throughput_and_length_metrics(batch, batch_time=2.0, max_output_tokens=100)

    assert metrics["inference.throughput/tokens_per_second"] == 144.5
    assert metrics["inference.throughput/requests_per_second"] == 1.5
    assert metrics["inference.length/response_tokens_total"] == 289
    assert metrics["inference.length/response_count"] == 3
    assert metrics["inference.length/response_tokens_max"] == 100
    assert metrics["inference.length/truncated_count"] == 1
    assert metrics["inference.length/truncated_rate"] == pytest.approx(1 / 3)
    assert metrics["inference.length/cap_tokens"] == 100
    assert metrics["inference.length/cap_saturation_threshold_tokens"] == 95.0
    assert metrics["inference.length/cap_saturated_count"] == 2
    assert metrics["inference.length/cap_saturated_rate"] == pytest.approx(2 / 3)


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


def test_sample_batch_attaches_schedule_assignment_metadata():
    curriculum_config = CurriculumConfig(
        lessons={
            "math_full": LessonConfig(
                lesson_id="math_full",
                env_config=EnvConfig(env_class="unused", env_args={}),
                sampling_params=SamplingParams(
                    n_prompts=3,
                    n_generations_per_prompt=2,
                    temperature=0.7,
                    top_k=5,
                ),
            )
        },
        max_seq_len=128,
    )
    env = _FakeEvalFiniteEnv(eval_len=4)
    inference_ctx = _FakeEvalInferenceContext()
    assignment = rollout_assignment(
        worker_index=1,
        lesson_id="math_full",
        worker_seed=1043,
        dataset_len=env.train_len(),
        start_position=0,
        n_examples=3,
    )

    worker = object.__new__(RolloutWorker)
    worker.config = SimpleNamespace(curriculum_config=curriculum_config, system_prompt=None, worker_index=1, seed=1043)
    worker._policy_ctx = inference_ctx
    worker._current_weight_step = 9
    worker._load_environment = lambda _lesson_id: env

    batch, _metrics = worker._sample_batch(
        lesson_id="math_full",
        n_examples=3,
        n_generations=2,
        mode="train",
        rng=0,
        indices=list(assignment.indices),
        assignment=assignment,
    )

    assert batch is not None
    assert inference_ctx.calls[0]["prompts"] == [f"train-{idx}" for idx in assignment.indices]
    assert batch.metadata.assignment_id == assignment.assignment_id
    assert batch.metadata.worker_index == 1
    assert batch.metadata.worker_seed == 1043
    assert batch.metadata.schedule_indices == assignment.indices
    assert batch.groups[0].rollouts[0].metadata.assignment_id == assignment.assignment_id


def test_reserve_and_commit_train_assignment_use_logical_worker_cursor():
    env = _FakeEvalFiniteEnv(eval_len=5)
    reserved = []
    committed = []

    class _RunState:
        reserve_rollout_assignment = _FakeRemoteMethod(
            lambda **kwargs: reserved.append(kwargs)
            or rollout_assignment(
                worker_index=kwargs["worker_index"],
                lesson_id=kwargs["lesson_id"],
                worker_seed=kwargs["worker_seed"],
                dataset_len=kwargs["dataset_len"],
                start_position=0,
                n_examples=kwargs["n_examples"],
            )
        )
        commit_rollout_assignment = _FakeRemoteMethod(lambda **kwargs: committed.append(kwargs))

    worker = object.__new__(RolloutWorker)
    worker.config = SimpleNamespace(worker_index=2, seed=1044)
    worker._runtime = SimpleNamespace(run_state=_RunState())
    worker._load_environment = lambda _lesson_id: env

    assignment = worker._reserve_train_assignment("math_full", n_examples=3)
    worker._commit_train_assignment(assignment)

    assert assignment is not None
    assert reserved == [
        {
            "worker_index": 2,
            "lesson_id": "math_full",
            "worker_seed": 1044,
            "dataset_len": 5,
            "n_examples": 3,
        }
    ]
    assert committed == [
        {
            "worker_index": 2,
            "lesson_id": "math_full",
            "assignment_id": assignment.assignment_id,
        }
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


def test_evaluate_curriculum_uses_configured_eval_modes_and_fixed_finite_order():
    update_lesson_stats = _FakeUpdateLessonStats()
    inference_ctx = _FakeEvalInferenceContext()
    eval_env = _FakeEvalFiniteEnv(eval_len=3)
    logged = []

    curriculum_config = CurriculumConfig(
        lessons={
            "math_full": LessonConfig(
                lesson_id="math_full",
                env_config=EnvConfig(env_class="unused", env_args={}),
                sampling_params=SamplingParams(
                    temperature=1.0,
                    top_k=4096,
                    n_prompts=2,
                    n_generations_per_prompt=4,
                    max_output_tokens=64,
                    stop_tokens=["</s>"],
                ),
            )
        },
        max_seq_len=128,
        eval_sampling_params=[
            EvalSamplingParams(
                name="pass1_greedy",
                n_examples=None,
                n_generations=1,
                temperature=0.0,
                top_k=-1,
                max_output_tokens=32,
                stop_tokens=["stop"],
                update_curriculum_stats=True,
            ),
            EvalSamplingParams(
                name="pass16_sample",
                n_examples=3,
                n_generations=16,
                temperature=1.0,
                top_k=4096,
                max_output_tokens=64,
                stop_tokens=["</s>"],
                update_curriculum_stats=False,
            ),
        ],
    )

    worker = object.__new__(RolloutWorker)
    worker.config = SimpleNamespace(curriculum_config=curriculum_config, system_prompt=None)
    worker._policy_ctx = inference_ctx
    worker._current_weight_step = 5
    worker._load_environment = lambda _lesson_id: eval_env
    worker._log_lesson_eval = lambda **kwargs: logged.append(kwargs)
    worker._curriculum_actor = SimpleNamespace(update_lesson_stats=update_lesson_stats)

    worker._evaluate_curriculum(rng=123, step=7)

    assert inference_ctx.calls == [
        {
            "prompts": ["eval-0", "eval-1", "eval-2"],
            "temperature": 0.0,
            "n": 1,
            "max_tokens": 32,
            "top_k": -1,
            "stop": ["stop"],
        },
        {
            "prompts": ["eval-0", "eval-1", "eval-2"],
            "temperature": 1.0,
            "n": 16,
            "max_tokens": 64,
            "top_k": 4096,
            "stop": ["</s>"],
        },
    ]
    assert [entry["eval_type"] for entry in logged] == ["eval_pass1_greedy", "eval_pass16_sample"]
    assert len(update_lesson_stats.calls) == 1


def test_build_prompt_example_metrics_is_deterministic_without_ambient_random(monkeypatch):
    monkeypatch.setattr("marin.rl.rollout_worker.random.sample", lambda *args, **kwargs: pytest.fail("random.sample"))
    monkeypatch.setattr("marin.rl.rollout_worker.random.choice", lambda *args, **kwargs: pytest.fail("random.choice"))

    class _Tokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            return ",".join(str(int(token_id)) for token_id in token_ids)

    worker = object.__new__(RolloutWorker)
    worker._tokenizer = _Tokenizer()
    batch = RolloutBatch(
        groups=[
            RolloutGroup(
                rollouts=[
                    Rollout(
                        env_name="fake",
                        env_example_id=f"eval_{idx}",
                        prompt_tokens=np.array([idx], dtype=np.int32),
                        response_tokens=np.array([idx + 10], dtype=np.int32),
                        response_logprobs=np.array([-0.1], dtype=np.float32),
                        token_rewards=np.array([1.0], dtype=np.float32),
                        episode_reward=1.0,
                        temperature=0.0,
                        top_k=-1,
                        is_truncated=False,
                    )
                ]
            )
            for idx in range(3)
        ],
        metadata=RolloutMetadata(),
    )

    metrics = worker._build_prompt_example_metrics("math_full", batch, step=5, eval_type="eval_pass1_greedy")

    assert "samples/p1g/math_full" in metrics


def test_sample_table_metric_key_fits_long_wandb_run_names():
    key = _sample_table_metric_key("eval_pass1_greedy", "math_full")
    long_run_id = "llama-3.1-8bi-math500-rlooIS-sched-100s-uc1-20260524-010114-rollout-0"

    # W&B derives table artifact names from the run id and metric key.
    artifact_name = f"run-{long_run_id}-{key.replace('/', '')}-abcdef"

    assert key == "samples/p1g/math_full"
    assert len(artifact_name) <= 128


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
