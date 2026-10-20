"""End-to-end CPU training through the production entrypoints on a tiny policy."""

import multiprocessing
import select
from contextlib import ExitStack
import os
import json
import sys
from multiprocessing.context import BaseContext
from pathlib import Path

import pytest
import ray
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


def _run_in_process(runs: BaseContext, target, *args, **kwargs) -> None:
    run = runs.Process(target=target, args=args, kwargs=kwargs)
    run.start()
    try:
        run.join(RUN_TIMEOUT_SECONDS)
        if run.exitcode is None:
            pytest.fail(f"the run did not finish within {RUN_TIMEOUT_SECONDS} seconds")
        assert run.exitcode == 0
    finally:
        if run.is_alive():
            run.kill()
            run.join(timeout=10)


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
    _run_in_process(
        runs,
        run_experiment,
        root,
        model,
        mode,
        shape,
        max_steps=steps,
        checkpoint_interval=checkpoint_interval,
        dump_data_batch=mode is TrainingMode.SYNC,
    )


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
    ("mode", "shape", "overlap"),
    [
        (TrainingMode.SYNC, RolloutShape.SINGLE_TURN, False),
        (TrainingMode.SYNC, RolloutShape.STEP_WISE, False),
        (TrainingMode.ASYNC, RolloutShape.STEP_WISE, True),
    ],
)
def test_tiny_policy_trains_to_max_steps(
    runs: BaseContext, tmp_path: Path, tiny_policy: Path, mode: TrainingMode, shape: RolloutShape, overlap: bool
):
    if overlap:
        _run_in_process(runs, _run_overlapping_training, tmp_path, tiny_policy, mode, shape)
    else:
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


ENGINE_START_FIFO = "engine-start"
POLICY_READY_FIFO = "policy-ready"


class _PolicyReadyWorker(CPUPolicyWorker):
    def init_model(self, model_path, num_training_steps=None):
        root = Path(self.cfg.trainer.export_path)
        with (root / ENGINE_START_FIFO).open("rb", buffering=0) as gate:
            assert gate.read(1) == b"e"
        super().init_model(model_path, num_training_steps)
        with (root / POLICY_READY_FIFO).open("wb", buffering=0) as gate:
            gate.write(b"p")


class _OverlappingTinyTrainingExp(experiment.TinyTrainingExp):
    def get_worker_classes(self):
        return ray.remote(num_gpus=1)(_PolicyReadyWorker), None, None

    def create_inference_engine_client(self, *, operation=experiment.EntrypointOperation.TRAIN):
        root = Path(self.cfg.trainer.export_path)
        with (root / ENGINE_START_FIFO).open("wb", buffering=0) as gate:
            gate.write(b"e")
        with (root / POLICY_READY_FIFO).open("rb", buffering=0) as gate:
            ready, _, _ = select.select([gate], [], [], RUN_TIMEOUT_SECONDS / 2)
            if not ready:
                raise TimeoutError("Policy did not initialize while engine startup waited")
            assert gate.read(1) == b"p"
        return super().create_inference_engine_client(operation=operation)


def _run_overlapping_training(root: Path, model: Path, mode: TrainingMode, shape: RolloutShape):
    cfg = experiment.tiny_training_config(
        root,
        model,
        mode,
        shape,
        max_steps=NUM_STEPS,
        checkpoint_interval=-1,
    )
    cfg.trainer.placement.policy_strict_spread_pg = True
    cfg.trainer.placement.overlap_init = True
    control_root = Path(cfg.trainer.export_path)
    control_root.mkdir(parents=True, exist_ok=True)
    for name in (ENGINE_START_FIFO, POLICY_READY_FIFO):
        os.mkfifo(control_root / name)
    with ExitStack() as gates:
        for name in (ENGINE_START_FIFO, POLICY_READY_FIFO):
            gates.callback(os.close, os.open(control_root / name, os.O_RDWR | os.O_NONBLOCK))
        experiment.run_with_exp(cfg, _OverlappingTinyTrainingExp)
