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
        return ray.remote(num_gpus=1)(_BarrierWorker), None


class _OverlappingWorker(CPUPolicyWorker):
    def init_model(self, *args, **kwargs):
        control = Path(self.cfg.trainer.export_path)
        with (control / "learner-loading").open("wb", buffering=0) as ready:
            ready.write(b"r")
        with (control / "inference-started").open("rb", buffering=0) as gate:
            assert select.select([gate], [], [], 60)[0], "inference startup waited for model loading"
            assert gate.read(1) == b"r"
        super().init_model(*args, **kwargs)


class _OverlappingExp(experiment.TinyTrainingExp):
    def get_worker_classes(self):
        return ray.remote(num_gpus=1)(_OverlappingWorker), None

    def create_inference_engine_client(self, **kwargs):
        control = Path(self.cfg.trainer.export_path)
        with (control / "learner-loading").open("rb", buffering=0) as gate:
            assert select.select([gate], [], [], 60)[0], "inference started before learner model loads were queued"
            assert gate.read(1) == b"r"
        with (control / "inference-started").open("wb", buffering=0) as ready:
            ready.write(b"r")
        return super().create_inference_engine_client(**kwargs)


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
    assert [record["trainer/global_step"] for record in read_metrics(tmp_path) if "policy/raw_grad_norm" in record] == [
        1
    ]


def _run_overlapping_training(root, model):
    cfg = tiny_training_config(
        root, model, TrainingMode.SYNC, RolloutShape.SINGLE_TURN, max_steps=1, checkpoint_interval=-1
    )
    cfg.trainer.placement.policy_strict_spread_pg = True
    control = Path(cfg.trainer.export_path)
    control.mkdir(parents=True, exist_ok=True)
    learner_loading = control / "learner-loading"
    inference_started = control / "inference-started"
    os.mkfifo(learner_loading)
    os.mkfifo(inference_started)
    # Keep both FIFOs open so select bounds the rendezvous even if a peer fails.
    with learner_loading.open("rb+", buffering=0), inference_started.open("rb+", buffering=0):
        experiment.run_with_exp(cfg, _OverlappingExp)


def test_disjoint_learner_and_inference_start_together(forkserver_context, tmp_path, tiny_policy):
    run = forkserver_context.Process(target=_run_overlapping_training, args=(tmp_path, tiny_policy))
    run.start()
    run.join(RUN_TIMEOUT_SECONDS)
    if run.exitcode is None:
        run.kill()
        run.join()
        pytest.fail("overlapping startup did not finish")
    assert run.exitcode == 0
    assert [record["trainer/global_step"] for record in read_metrics(tmp_path) if "policy/raw_grad_norm" in record] == [
        1
    ]
