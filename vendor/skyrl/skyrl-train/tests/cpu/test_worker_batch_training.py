import json
import multiprocessing
import sys
from multiprocessing.context import BaseContext
from pathlib import Path

import pytest
import torch
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.importance_ratio_diagnostics import MISMATCH_RATIO_HISTOGRAM_LOG_WIDTH
from skyrl_train.entrypoints.main_base import BasePPOExp

from tests.cpu.tiny_training import experiment
from tests.cpu.tiny_training.experiment import RolloutShape, TrainingMode, read_metrics
from tests.cpu.tiny_training.tiny_model import build_tiny_policy

pytestmark = [pytest.mark.slow, pytest.mark.timeout(900)]


RUN_TIMEOUT_SECONDS = 300


@pytest.fixture(scope="module")
def tiny_policy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_policy(tmp_path_factory.mktemp("tiny_policy"))


@pytest.fixture(scope="module")
def runs() -> BaseContext:
    if sys.platform == "darwin":
        # Forkserver preload clones initialized macOS proxy/preferences libraries; use a fresh interpreter.
        return multiprocessing.get_context("spawn")
    context = multiprocessing.get_context("forkserver")
    context.set_forkserver_preload([experiment.__name__])
    return context


def _trained_steps(root: Path) -> list[dict]:
    return [record for record in read_metrics(root) if "policy/raw_grad_norm" in record]


def _worker_batch_config(root: Path, tiny_policy: Path, builder: str, estimator: str = "rloo_n"):
    cfg = experiment.tiny_training_config(
        root,
        tiny_policy,
        TrainingMode.SYNC,
        RolloutShape.SINGLE_TURN,
        max_steps=2,
        checkpoint_interval=-1,
        dp_size=2,
        micro_batch_size=3,
        dump_data_batch=builder != "worker",
    )
    cfg.trainer.batch_builder = builder
    cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 3
    cfg.trainer.training_metrics = True
    cfg.trainer.algorithm.advantage_estimator = estimator
    cfg.trainer.algorithm.group_advantage_min_size = 2 if estimator == "rloo_n" else None
    cfg.generator.trajectory_reward_shaping.enabled = True
    cfg.generator.trajectory_reward_shaping.overlong.penalty_scale = 0.0
    return cfg


class PreferenceTrainingExp(experiment.TinyTrainingExp):
    get_train_dataset = BasePPOExp.get_train_dataset
    create_inference_engine_client = BasePPOExp.create_inference_engine_client


def run_preference_training(cfg):
    experiment.run_with_exp(cfg, PreferenceTrainingExp)


def preference_config(root: Path, model: Path):
    cfg = _worker_batch_config(root, model, "verify", "uniform")
    path = root / "pairs.jsonl"
    rows = []
    for index in range(8):
        rows.append(
            {
                "chosen_input_ids": [10, 11, 20 + index, 30, 31, 40],
                "chosen_assistant_masks": [0, 0, 1, 0, 1, 1],
                "rejected_input_ids": [10, 11, 45 + index, 32, 33],
                "rejected_assistant_masks": [0, 0, 1, 0, 1],
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    cfg.data.train_data = [str(path)]
    cfg.data.val_data = []
    cfg.data.preference_pair_format = "tokenized"
    cfg.environment.env_class = "preference_pair"
    cfg.generator.n_samples_per_prompt = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.algorithm.policy_loss_type = "sft"
    cfg.trainer.algorithm.off_policy_correction = "none"
    cfg.generator.trajectory_reward_shaping.enabled = False
    return cfg


# Every remaining metric key must have a producer in both source implementations.
# The driver publishes exact p95/p999; workers publish the shared histogram p99.
_SOURCE_METRIC_KEY_DIFFERENCES = frozenset(
    {
        "policy/mismatch/pooled/log_ratio_abs_p95",
        "policy/mismatch/pooled/log_ratio_abs_p999",
        "policy/mismatch/staleness0/log_ratio_abs_p95",
        "policy/mismatch/staleness0/log_ratio_abs_p999",
        "timing/assemble_generation_group_mini_batch",
        "timing/postprocess_trajectory_batch",
        "timing/convert_to_training_input",
        "timing/dump_data_batch",
        "timing/load_worker_batch",
        "timing/verify_driver_preparation",
        "timing/verify_driver_forward",
        "timing/verify_driver_finalization",
        "timing/compute_advantages_and_returns",
        "timing/prepare_worker_training_input",
    }
)


def _run_worker_batch_case(runs: BaseContext, cfg, target=experiment.run_tiny_training):
    run = runs.Process(target=target, args=(cfg,))
    run.start()
    run.join(RUN_TIMEOUT_SECONDS)
    if run.exitcode is None:
        run.kill()
        run.join()
        pytest.fail("worker verification did not finish")
    assert run.exitcode == 0


@pytest.fixture(scope="module")
def driver_batch_metrics(runs, tmp_path_factory, tiny_policy):
    baselines = {}

    def for_estimator(estimator):
        if estimator not in baselines:
            root = tmp_path_factory.mktemp(f"driver_batch_metrics_{estimator}")
            _run_worker_batch_case(runs, _worker_batch_config(root, tiny_policy, "driver", estimator))
            baselines[estimator] = _trained_steps(root)
        return baselines[estimator]

    return for_estimator


@pytest.mark.parametrize(
    ("builder", "estimator"),
    [("verify", "rloo_n"), ("worker", "rloo_n"), ("verify", "grpo")],
    ids=["verify", "worker", "verify-grpo"],
)
def test_worker_batches_verify_two_tis_steps_with_a_group_split_between_dp_ranks(
    runs: BaseContext, tmp_path: Path, tiny_policy: Path, builder: str, estimator: str, driver_batch_metrics
):
    _run_worker_batch_case(runs, _worker_batch_config(tmp_path, tiny_policy, builder, estimator))
    steps = _trained_steps(tmp_path)
    baseline = driver_batch_metrics(estimator)
    assert [record["trainer/global_step"] for record in steps] == [1, 2]
    assert all(record["policy/raw_grad_norm"] > 0 for record in steps)
    assert [set(record) - _SOURCE_METRIC_KEY_DIFFERENCES for record in steps] == [
        set(record) - _SOURCE_METRIC_KEY_DIFFERENCES for record in baseline
    ]
    for actual, expected in zip(steps, baseline, strict=True):
        for key in expected:
            if (
                key.startswith(
                    (
                        "generate/",
                        "reward/",
                        "loss/",
                        "async/staleness",
                        "policy/mismatch/",
                        "policy/correction/",
                        "consumed/",
                        "policy/rollout_train_prob_diff",
                    )
                )
                and key not in _SOURCE_METRIC_KEY_DIFFERENCES
            ):
                if key.endswith("log_ratio_abs_p99"):
                    assert abs(actual[key] - expected[key]) <= MISMATCH_RATIO_HISTOGRAM_LOG_WIDTH, key
                else:
                    assert actual[key] == pytest.approx(expected[key], rel=1e-6, abs=1e-7, nan_ok=True), key
    if builder == "worker":
        assert all(record["generate/avg_num_tokens"] > 0 for record in steps)
        return
    for step in (1, 2):
        batch = TrainingInputBatch().load(tmp_path / f"exports/dumped_data/global_step_{step}_training_input.pkl")
        assert batch.batch_size == 12
        lengths = batch["response_mask"].sum(-1)
        assert torch.unique(lengths).numel() > 1
        assert torch.unique(batch["advantages"]).numel() > 2
        eligible = batch["loss_mask"].bool()
        assert "correction_weights" in batch
        assert (batch["correction_weights"][eligible] != 1).any()


def test_verify_reads_shared_object_payloads_for_two_async_steps(runs, tmp_path, tiny_policy):
    cfg = experiment.tiny_training_config(
        tmp_path,
        tiny_policy,
        TrainingMode.ASYNC,
        RolloutShape.SINGLE_TURN,
        max_steps=2,
        checkpoint_interval=1,
        dp_size=2,
        micro_batch_size=4,
        dump_data_batch=True,
    )
    cfg.trainer.batch_builder = "verify"
    _run_worker_batch_case(runs, cfg)
    steps = _trained_steps(tmp_path)
    assert [row["trainer/global_step"] for row in steps] == [1, 2]
    assert all(row["policy/raw_grad_norm"] > 0 for row in steps)
    assert list((tmp_path / "rollouts").rglob("*.pkl"))
    for step in (1, 2):
        batch = TrainingInputBatch().load(tmp_path / f"exports/dumped_data/global_step_{step}_training_input.pkl")
        assert batch.batch_size == 16
        assert batch["action_log_probs"].isfinite().all()

    cfg.trainer.max_steps = 3
    _run_worker_batch_case(runs, cfg)
    assert [row["trainer/global_step"] for row in _trained_steps(tmp_path)] == [1, 2, 3]


def test_verify_sft_uniform_preserves_preference_assistant_masks(runs, tmp_path, tiny_policy):
    _run_worker_batch_case(runs, preference_config(tmp_path, tiny_policy), target=run_preference_training)
    steps = _trained_steps(tmp_path)
    assert [row["trainer/global_step"] for row in steps] == [1, 2]
    assert all(row["policy/raw_grad_norm"] > 0 for row in steps)
    for step in (1, 2):
        batch = TrainingInputBatch().load(tmp_path / f"exports/dumped_data/global_step_{step}_training_input.pkl")
        assert batch.batch_size == 6
        torch.testing.assert_close(batch["advantages"], torch.ones_like(batch["advantages"]), rtol=0, atol=0)
        torch.testing.assert_close(batch["returns"], torch.ones_like(batch["returns"]), rtol=0, atol=0)
        assert batch["response_mask"].sum(-1).unique().numel() == 2
        assert ((batch["loss_mask"] == 0) & (batch["response_mask"] > 0)).any()
        assert batch["pair_roles"].tolist() == [1, -1, 1, -1, 1, -1]
