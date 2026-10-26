from contextlib import ExitStack
import os
import select
from pathlib import Path

import pytest
import ray

from tests.cpu.tiny_training import experiment
from tests.cpu.tiny_training.cpu_backend import CPUPolicyWorker
from tests.cpu.tiny_training.experiment import RolloutShape, TrainingMode, read_metrics, tiny_training_config
from tests.cpu.tiny_training.tiny_model import build_tiny_policy

pytestmark = pytest.mark.slow
RUN_TIMEOUT_SECONDS = 300


@pytest.fixture(scope="module")
def tiny_policy(tmp_path_factory):
    return build_tiny_policy(tmp_path_factory.mktemp("tiny_policy"))


class _BarrierWorker(CPUPolicyWorker):
    def __init__(self, cfg, rank, **kwargs):
        super().__init__(cfg=cfg, rank=rank, **kwargs)
        with (Path(cfg.trainer.export_path) / "rank-ready").open("rb" if rank == 0 else "wb", buffering=0) as gate:
            if rank == 0:
                assert select.select([gate], [], [], 20)[0], "rank 1 constructor never started"
                assert gate.read(1) == b"r"
            else:
                gate.write(b"r")


class _BarrierExp(experiment.TinyTrainingExp):
    def get_worker_classes(self):
        return ray.remote(num_gpus=1)(_BarrierWorker), None, None


def _run_barrier_training(root, model):
    cfg = tiny_training_config(
        root, model, TrainingMode.SYNC, RolloutShape.SINGLE_TURN, max_steps=1, checkpoint_interval=-1, dp_size=2
    )
    control = Path(cfg.trainer.export_path)
    control.mkdir(parents=True, exist_ok=True)
    os.mkfifo(control / "rank-ready")
    descriptor = os.open(control / "rank-ready", os.O_RDWR | os.O_NONBLOCK)
    try:
        experiment.run_with_exp(cfg, _BarrierExp)
    finally:
        os.close(descriptor)


def test_all_rank_constructors_start_before_process_group_init(forkserver_context, tmp_path, tiny_policy):
    run = forkserver_context.Process(target=_run_barrier_training, args=(tmp_path, tiny_policy))
    run.start()
    run.join(RUN_TIMEOUT_SECONDS)
    if run.exitcode is None:
        run.kill()
        run.join()
        pytest.fail("two-rank startup did not finish")
    assert run.exitcode == 0
    assert [
        record["trainer/global_step"] for record in read_metrics(tmp_path) if "policy/raw_grad_norm" in record
    ] == [1]


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


def _run_overlapping_training(root: Path, model: Path) -> None:
    cfg = experiment.tiny_training_config(
        root, model, TrainingMode.SYNC, RolloutShape.SINGLE_TURN, max_steps=1, checkpoint_interval=-1
    )
    cfg.trainer.placement.policy_strict_spread_pg = True
    cfg.trainer.placement.overlap_init = True
    control_root = Path(cfg.trainer.export_path)
    control_root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as gates:
        for name in (ENGINE_START_FIFO, POLICY_READY_FIFO):
            os.mkfifo(control_root / name)
            gates.callback(os.close, os.open(control_root / name, os.O_RDWR | os.O_NONBLOCK))
        experiment.run_with_exp(cfg, _OverlappingTinyTrainingExp)


def test_overlap_init_builds_policy_while_engines_start(forkserver_context, tmp_path, tiny_policy):
    run = forkserver_context.Process(target=_run_overlapping_training, args=(tmp_path, tiny_policy))
    run.start()
    run.join(RUN_TIMEOUT_SECONDS)
    if run.exitcode is None:
        run.kill()
        run.join()
        pytest.fail(f"the run did not finish within {RUN_TIMEOUT_SECONDS} seconds")
    assert run.exitcode == 0
    trained = [record for record in read_metrics(tmp_path) if "policy/raw_grad_norm" in record]
    assert [record["trainer/global_step"] for record in trained] == [1]
