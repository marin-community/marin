"""End-to-end CPU training through the production entrypoints on a tiny policy."""

import multiprocessing
import asyncio
import os
import signal
import threading
import time
import psutil
import ray
from multiprocessing.connection import Connection
import json
import sys
from multiprocessing.context import BaseContext
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM
from omegaconf import OmegaConf
from skyrl_train.callbacks.base import TrainerCallback
from skyrl_train.callbacks.builtin import register_callback

from marinskyrl.checkpoint_paths import POLICY_CHECKPOINT_SUBDIRECTORY
from tests.cpu.tiny_training.cpu_backend import CHECKPOINT_FILE_TEMPLATE, CausalLMPolicy, CPUPolicyWorker
from tests.cpu.tiny_training.fixed_batch import fixed_training_batch, run_fixed_update
from skyrl_train.rollouts.payloads import ROLLOUT_OBJECT_SUFFIX
from skyrl_train.training_batch import TrainingInputBatch
from tests.cpu.tiny_training import experiment
from tests.cpu.tiny_training.experiment import (
    MAX_STALENESS_STEPS,
    N_SAMPLES_PER_PROMPT,
    OBJECT_STORE_PAYLOADS,
    TRAIN_BATCH_SIZE,
    RolloutShape,
    TrainingMode,
    read_metrics,
    run_experiment,
    sampling_kind,
)
from tests.cpu.tiny_training.tiny_model import CURRICULUM_BINS, build_tiny_policy

pytestmark = pytest.mark.slow

NUM_STEPS = 3
RESUMED_STEP = 1
# A run takes under half a minute on an idle host. The margin absorbs slower CI hosts and concurrent pytest-xdist
# workers, and stays above the experiment's admission stall timeout so a stall reports its own error.
RUN_TIMEOUT_SECONDS = 300


@register_callback("test_step_limit")
class _StepLimit(TrainerCallback):
    def __init__(self, limit: int):
        self.limit = limit

    def on_train_begin(self, state, control, **kwargs):
        control.step_limit = self.limit
        trainer = kwargs["trainer"]
        path = Path(trainer.cfg.trainer.export_path) / f"start-{state.global_step}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"checkpoint_path": trainer.loaded_checkpoint_path}))
        return control


def _run_limited_training(root: Path, model: Path, limits: tuple[int, ...]) -> None:
    cfg = experiment.tiny_training_config(
        root, model, TrainingMode.SYNC, RolloutShape.SINGLE_TURN, max_steps=1, checkpoint_interval=1
    )
    OmegaConf.update(
        cfg,
        "trainer.callbacks",
        [{"type": "test_step_limit", "limit": limit} for limit in limits] + [{"type": "checkpoint", "save_steps": 1}],
        force_add=True,
    )
    experiment.run_tiny_training(cfg)


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


def _train(
    runs: BaseContext,
    root: Path,
    model: Path,
    mode: TrainingMode,
    shape: RolloutShape,
    *,
    steps: int,
    checkpoint_interval: int = -1,
) -> None:
    run = runs.Process(
        target=run_experiment,
        args=(root, model, mode, shape),
        kwargs={
            "max_steps": steps,
            "checkpoint_interval": checkpoint_interval,
            "dump_data_batch": mode is TrainingMode.SYNC,
        },
    )
    run.start()
    run.join(RUN_TIMEOUT_SECONDS)
    if run.exitcode is None:
        run.kill()
        run.join()
        pytest.fail(f"the run did not finish within {RUN_TIMEOUT_SECONDS} seconds")
    assert run.exitcode == 0


def _trained_steps(root: Path) -> list[dict]:
    return [record for record in read_metrics(root) if "policy/raw_grad_norm" in record]


def _assert_trained_to_max_steps(root: Path, mode: TrainingMode, shape: RolloutShape) -> None:
    steps = _trained_steps(root)
    assert [record["trainer/global_step"] for record in steps] == list(range(1, NUM_STEPS + 1))
    # A batch whose samples all score alike, which happens by chance, leaves no advantage to train on.
    assert any(record["policy/raw_grad_norm"] > 0 for record in steps)
    assert max(record["async/staleness_max"] for record in steps) <= MAX_STALENESS_STEPS[mode]
    if mode is TrainingMode.SYNC:
        for step in range(1, NUM_STEPS + 1):
            batch = TrainingInputBatch().load(root / f"exports/dumped_data/global_step_{step}_training_input.pkl")
            eligible = batch["loss_mask"].bool()
            expected = torch.zeros_like(batch["action_log_probs"], dtype=torch.float32)
            expected[eligible] = (
                (batch["action_log_probs"][eligible].float() - batch["rollout_logprobs"][eligible].float())
                .exp()
                .clamp(max=2)
            )
            torch.testing.assert_close(batch["correction_weights"], expected)
            assert steps[step - 1]["policy/correction/weight_mean"] == pytest.approx(
                expected.sum().item() / eligible.sum().item()
            )
    if shape is RolloutShape.STEP_WISE:
        # A trajectory that answers wrongly on its first turn trains one sample per turn.
        trajectories = NUM_STEPS * TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT
        assert sum(record["consumed/sequences"] for record in steps) > trajectories
    if sampling_kind(mode, shape) is not None:
        # Without dynamic sampling, the groups each step judged are exactly its batch.
        for record in steps:
            assert sum(record[f"curriculum/{name}/groups"] for name in CURRICULUM_BINS) == TRAIN_BATCH_SIZE
    if OBJECT_STORE_PAYLOADS[mode]:
        # Every trained group was read back from its object, and the objects stay for groups generated but not
        # trained.
        objects = list((root / "rollouts").glob(f"*{ROLLOUT_OBJECT_SUFFIX}"))
        assert len(objects) >= NUM_STEPS * TRAIN_BATCH_SIZE


# The resume test covers asynchronous single-turn training.
@pytest.mark.parametrize(
    ("mode", "shape"),
    [
        (TrainingMode.SYNC, RolloutShape.SINGLE_TURN),
        (TrainingMode.SYNC, RolloutShape.STEP_WISE),
        (TrainingMode.ASYNC, RolloutShape.STEP_WISE),
    ],
)
def test_tiny_policy_trains_to_max_steps(
    runs: BaseContext, tmp_path: Path, tiny_policy: Path, mode: TrainingMode, shape: RolloutShape
):
    _train(runs, tmp_path, tiny_policy, mode, shape, steps=NUM_STEPS)

    _assert_trained_to_max_steps(tmp_path, mode, shape)


def test_async_training_resumes_with_committed_groups(runs: BaseContext, tmp_path: Path, tiny_policy: Path):
    mode, shape = TrainingMode.ASYNC, RolloutShape.SINGLE_TURN
    _train(runs, tmp_path, tiny_policy, mode, shape, steps=RESUMED_STEP, checkpoint_interval=1)
    # The first two batches' leases open together, so the second batch's groups commit while the first trains
    # and its checkpoint holds them. A later step's leases open only when its batch is taken, and whether they
    # commit before that step's checkpoint is a race.
    state = torch.load(tmp_path / "ckpts" / f"global_step_{RESUMED_STEP}" / "data.pt", weights_only=False)
    assert state.ready

    _train(runs, tmp_path, tiny_policy, mode, shape, steps=NUM_STEPS, checkpoint_interval=1)

    _assert_trained_to_max_steps(tmp_path, mode, shape)


def test_callback_limits_resume_at_max_steps_and_keep_the_smallest_limit(runs, tmp_path, tiny_policy):
    for limits in ((1,), (3, 4), (3,)):
        run = runs.Process(target=_run_limited_training, args=(tmp_path, tiny_policy, limits))
        run.start()
        run.join(RUN_TIMEOUT_SECONDS)
        if run.exitcode is None:
            run.kill()
            run.join()
            pytest.fail("callback-limited training did not finish")
        assert run.exitcode == 0
    assert [record["trainer/global_step"] for record in _trained_steps(tmp_path)] == [1, 2, 3]
    checkpoint = tmp_path / "ckpts" / "global_step_3" / "trainer_state.pt"
    assert checkpoint.exists()
    for step in (1, 3):
        provenance = json.loads((tmp_path / "exports" / f"start-{step}.json").read_text())
        assert provenance["checkpoint_path"] == str(tmp_path / "ckpts" / f"global_step_{step}")


def test_one_step_is_independent_of_micro_batch_size(runs: BaseContext, tmp_path: Path, tiny_policy: Path):
    batch = fixed_training_batch(str(tiny_policy))
    model = CausalLMPolicy(AutoModelForCausalLM.from_pretrained(tiny_policy, dtype=torch.float32))
    log_probs = model(batch["sequences"], num_actions=4, attention_mask=batch["attention_mask"])
    ratios = (log_probs - batch["action_log_probs"]).exp()
    loss = (
        -(ratios * batch["advantages"] * batch["correction_weights"] * batch["loss_mask"]).sum()
        / batch["loss_mask"].sum()
    )
    loss.backward()
    reference_norm = torch.linalg.vector_norm(
        torch.stack([parameter.grad.norm() for parameter in model.parameters() if parameter.grad is not None])
    ).item()
    roots = [tmp_path / "micro_8", tmp_path / "micro_4"]
    for root, micro_batch_size in zip(roots, [8, 4], strict=True):
        run = runs.Process(target=run_fixed_update, args=(root, tiny_policy, micro_batch_size))
        run.start()
        run.join(RUN_TIMEOUT_SECONDS)
        if run.exitcode is None:
            run.kill()
            run.join()
            pytest.fail(f"the update did not finish within {RUN_TIMEOUT_SECONDS} seconds")
        assert run.exitcode == 0
    norms = [torch.load(root / "train_status.pt", weights_only=False)["raw_grad_norm"] for root in roots]
    for norm in norms:
        torch.testing.assert_close(norm, reference_norm, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(norms[0], norms[1], rtol=1e-5, atol=1e-6)
    for rank in range(2):
        states = [
            torch.load(
                root
                / "ckpts/global_step_1"
                / POLICY_CHECKPOINT_SUBDIRECTORY
                / CHECKPOINT_FILE_TEMPLATE.format(rank=rank),
                weights_only=False,
            )
            for root in roots
        ]
        for key, tensor in states[0]["model"].items():
            torch.testing.assert_close(tensor, states[1]["model"][key], rtol=1e-5, atol=1e-6, msg=key)


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


def _run_worker_batch_case(runs: BaseContext, cfg):
    run = runs.Process(target=experiment.run_tiny_training, args=(cfg,))
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
    if estimator == "grpo":
        assert [record["reward/zero_std_group_fraction"] for record in steps] == [
            record["reward/zero_std_group_fraction"] for record in baseline
        ]
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
        expected = torch.zeros_like(batch["action_log_probs"], dtype=torch.float32)
        expected[eligible] = (
            (batch["action_log_probs"][eligible] - batch["rollout_logprobs"][eligible]).exp().clamp(max=2)
        )
        torch.testing.assert_close(batch["correction_weights"], expected, rtol=0, atol=0)
        assert (expected[eligible] != 1).any()


_FORWARD_GATE_NAME = "tiny_training_forward_gate"


class ForwardFailure(RuntimeError):
    pass


class _ForwardGate:
    def __init__(self):
        self.ranks = set()
        self.both_entered = asyncio.Event()
        self.released = asyncio.Event()

    async def enter(self, rank: int):
        self.ranks.add(rank)
        if self.ranks == {0, 1}:
            self.both_entered.set()
        await self.both_entered.wait()

    async def wait_for_both(self):
        await self.both_entered.wait()
        return tuple(sorted(self.ranks))

    async def withhold(self):
        await self.released.wait()

    async def release(self):
        self.released.set()


class _FailedForwardWorker(CPUPolicyWorker):
    def forward_loaded(self, batch_id):
        gate = ray.get_actor(_FORWARD_GATE_NAME)
        ray.get(gate.enter.remote(self.mesh_rank.dp))
        if self.mesh_rank.dp == 0:
            raise ForwardFailure("injected policy forward failure")
        # A synchronous hold prevents this actor from servicing queued cleanup RPCs.
        ray.get(gate.withhold.remote())
        return super().forward_loaded(batch_id)


class _FailedForwardExp(experiment.TinyTrainingExp):
    def get_worker_classes(self):
        return ray.remote(num_gpus=1)(_FailedForwardWorker), None, None

    def get_trainer(self, *args, **kwargs):
        self.trainer = super().get_trainer(*args, **kwargs)
        return self.trainer


def _run_failed_forward(cfg, ready: Connection, log: Path):
    os.setsid()
    ready.send(("session", os.getpid()))
    with log.open("w") as output:
        os.dup2(output.fileno(), 1)
        os.dup2(output.fileno(), 2)
        experiment.validate_cfg(cfg)
        experiment.validate_rollout_launch(cfg, experiment.EntrypointOperation.TRAIN)
        gate = None
        try:
            ray.init(
                num_cpus=experiment.LOGICAL_CPUS,
                num_gpus=experiment.LOGICAL_GPUS,
                runtime_env={"env_vars": experiment.WORKER_ENV_VARS},
                include_dashboard=False,
            )
            gate = ray.remote(num_cpus=0)(_ForwardGate).options(name=_FORWARD_GATE_NAME).remote()

            def report_ready():
                ready.send(("forward_ranks", ray.get(gate.wait_for_both.remote())))

            threading.Thread(target=report_ready, daemon=True).start()
            exp = _FailedForwardExp(cfg)
            with pytest.raises(ray.exceptions.RayTaskError) as raised:
                exp.run()
            assert isinstance(raised.value.as_instanceof_cause(), ForwardFailure)
            assert len(exp.trainer.policy_model.actor_infos) == 2
            for actor in exp.trainer.policy_model.actor_infos:
                with pytest.raises(ray.exceptions.RayActorError):
                    ray.get(actor.handle.get_mesh_rank.remote(), timeout=10)
            metrics = Path(cfg.trainer.export_path) / experiment.METRICS_FILE
            assert not metrics.exists() or not _trained_steps(metrics.parent.parent)
        finally:
            try:
                if gate is not None:
                    ray.get(gate.release.remote(), timeout=10)
            finally:
                ray.shutdown()
                ready.close()


def test_worker_forward_failure_preserves_error_and_stops_policy_actors(runs, tmp_path, tiny_policy):
    cfg = _worker_batch_config(tmp_path, tiny_policy, "worker")
    cfg.trainer.max_steps = 1
    log = tmp_path / "failed-forward.log"
    ready, child_ready = runs.Pipe(duplex=False)
    run = runs.Process(target=_run_failed_forward, args=(cfg, child_ready, log))
    run.start()
    child_ready.close()
    session_started = False
    descendants = set()
    setup_deadline = time.monotonic() + RUN_TIMEOUT_SECONDS
    try:
        assert ready.poll(RUN_TIMEOUT_SECONDS), "failed-forward child did not start"
        assert ready.recv() == ("session", run.pid)
        session_started = True
        assert ready.poll(max(0, setup_deadline - time.monotonic())), "policy ranks did not reach forward"
        assert ready.recv() == ("forward_ranks", (0, 1))
        descendants.update(psutil.Process(run.pid).children(recursive=True))
        run.join(RUN_TIMEOUT_SECONDS)
        assert run.exitcode == 0, f"failed-forward training did not finish cleanly; exitcode={run.exitcode}"
    finally:
        if run.exitcode != 0:
            if run.is_alive():
                try:
                    descendants.update(psutil.Process(run.pid).children(recursive=True))
                    if session_started:
                        os.killpg(run.pid, signal.SIGKILL)
                    else:
                        run.kill()
                except (ProcessLookupError, psutil.NoSuchProcess):
                    pass
            for process in descendants:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(descendants, timeout=10)
            run.join(10)
            assert not alive and not run.is_alive(), "failed-forward child processes survived cleanup"
        ready.close()
        if run.exitcode != 0 and log.exists():
            print(log.read_text()[-16000:])
