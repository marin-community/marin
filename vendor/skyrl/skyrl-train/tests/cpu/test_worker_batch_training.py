import json
import multiprocessing
import sys
from multiprocessing.context import BaseContext
from pathlib import Path

import pytest
import torch
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


def _worker_batch_config(root: Path, tiny_policy: Path, builder: str, estimator: str = "grpo"):
    cfg = experiment.tiny_training_config(
        root,
        tiny_policy,
        TrainingMode.SYNC,
        RolloutShape.SINGLE_TURN,
        max_steps=2,
        checkpoint_interval=-1,
        dp_size=2,
        micro_batch_size=3,
        dump_data_batch=False,
    )
    cfg.trainer.batch_builder = builder
    # Serial rollout admission gives both builders the same row order.
    cfg.trajectory_runner.rollout_workers.num_workers = 1
    cfg.trainer.rollout_buffer.max_in_flight = 1
    cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 3
    cfg.trainer.training_metrics = True
    cfg.trainer.algorithm.advantage_estimator = estimator
    cfg.trainer.algorithm.group_advantage_min_size = None
    cfg.generator.trajectory_reward_shaping.enabled = True
    cfg.generator.trajectory_reward_shaping.overlong.penalty_scale = 0.0
    return cfg


class PreferenceTrainingExp(experiment.TinyTrainingExp):
    get_train_dataset = BasePPOExp.get_train_dataset
    create_inference_engine_client = BasePPOExp.create_inference_engine_client


def run_preference_training(cfg):
    experiment.run_with_exp(cfg, PreferenceTrainingExp)


def preference_config(root: Path, model: Path, builder: str):
    cfg = _worker_batch_config(root, model, builder, "uniform")
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


def _run_worker_batch_case(runs: BaseContext, cfg, target=experiment.run_tiny_training):
    run = runs.Process(target=target, args=(cfg,))
    run.start()
    run.join(RUN_TIMEOUT_SECONDS)
    if run.exitcode is None:
        run.kill()
        run.join()
        pytest.fail("worker training did not finish")
    assert run.exitcode == 0


@pytest.fixture(scope="module")
def driver_batch_metrics(runs, tmp_path_factory, tiny_policy):
    baselines = {}

    def for_case(case, config_builder, target):
        if case not in baselines:
            root = tmp_path_factory.mktemp(f"driver_batch_metrics_{case}")
            _run_worker_batch_case(runs, config_builder(root, tiny_policy, "driver"), target=target)
            baselines[case] = _trained_steps(root)
        return baselines[case]

    return for_case


@pytest.mark.parametrize(
    ("case", "config_builder", "target"),
    [
        ("grpo", _worker_batch_config, experiment.run_tiny_training),
        ("sft", preference_config, run_preference_training),
    ],
    ids=["grpo", "sft"],
)
def test_worker_batches_match_two_driver_updates(
    runs: BaseContext, tmp_path: Path, tiny_policy: Path, case, config_builder, target, driver_batch_metrics
):
    _run_worker_batch_case(runs, config_builder(tmp_path, tiny_policy, "worker"), target=target)
    steps = _trained_steps(tmp_path)
    baseline = driver_batch_metrics(case, config_builder, target)
    assert [row["trainer/global_step"] for row in steps] == [1, 2]
    assert all(row["policy/raw_grad_norm"] > 0 for row in steps)
    required = {"policy/policy_loss", "policy/raw_grad_norm"}
    absent = (
        set()
        if case == "sft"
        else {
            f"policy/mismatch/{scope}/{key}"
            for scope in ("pooled", "staleness0")
            for key in (
                "log_ratio_abs_p95",
                "log_ratio_abs_p99",
                "log_ratio_abs_p999",
                "frac_below_1e_5",
                "ess_fraction",
                "kl_k1",
                "kl_k3",
                "chi2",
            )
        }
        | {"policy/rollout_train_prob_diff_std"}
    )
    execution_prefixes = ("timing/", "perf/", "async/performance/", "system/")
    for actual, expected in zip(steps, baseline, strict=True):
        assert {key for key in expected.keys() - actual.keys() if not key.startswith(execution_prefixes)} == absent
        for key in required:
            assert actual[key] == expected[key], (key, actual[key], expected[key])
        for key in (actual.keys() & expected.keys()) - required:
            if key.startswith(execution_prefixes):
                continue
            assert actual[key] == pytest.approx(expected[key], rel=1e-6, abs=1e-7, nan_ok=True), key
    assert all(row["generate/avg_num_tokens"] > 0 for row in steps)


def test_worker_reads_shared_object_payloads_and_trains_after_resume(runs, tmp_path, tiny_policy):
    cfg = experiment.tiny_training_config(
        tmp_path,
        tiny_policy,
        TrainingMode.ASYNC,
        RolloutShape.SINGLE_TURN,
        max_steps=2,
        checkpoint_interval=1,
        dp_size=2,
        micro_batch_size=4,
        dump_data_batch=False,
    )
    cfg.trainer.batch_builder = "worker"
    _run_worker_batch_case(runs, cfg)
    assert [row["trainer/global_step"] for row in _trained_steps(tmp_path)] == [1, 2]
    assert list((tmp_path / "rollouts").rglob("*.pkl"))
    state = torch.load(tmp_path / "ckpts/global_step_2/data.pt", weights_only=False)
    assert any(group.verdict.trainable for group in state.ready)
    assert all(group.verdict.row_facts is None for group in state.ready)
    cfg.trainer.max_steps = 3
    _run_worker_batch_case(runs, cfg)
    steps = _trained_steps(tmp_path)
    assert [row["trainer/global_step"] for row in steps] == [1, 2, 3]
