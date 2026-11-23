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
OVERLAP_TIMEOUT_SECONDS = 60
LEARNER_LOADING = "learner-loading"
INFERENCE_STARTED = "inference-started"
READY = b"r"


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
        with (control / LEARNER_LOADING).open("wb", buffering=0) as ready:
            ready.write(READY)
        with (control / INFERENCE_STARTED).open("rb", buffering=0) as gate:
            assert select.select([gate], [], [], OVERLAP_TIMEOUT_SECONDS)[0], "inference startup stalled"
        super().init_model(*args, **kwargs)


class _OverlappingExp(experiment.TinyTrainingExp):
    def get_worker_classes(self):
        return ray.remote(num_gpus=1)(_OverlappingWorker), None

    def create_inference_engine_client(self, **kwargs):
        control = Path(self.cfg.trainer.export_path)
        with (control / LEARNER_LOADING).open("rb", buffering=0) as gate:
            assert select.select([gate], [], [], OVERLAP_TIMEOUT_SECONDS)[0], "learner did not start model loading"
        with (control / INFERENCE_STARTED).open("wb", buffering=0) as ready:
            ready.write(READY)
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


def _run_overlapping_training(root, model):
    cfg = tiny_training_config(
        root, model, TrainingMode.SYNC, RolloutShape.SINGLE_TURN, max_steps=1, checkpoint_interval=-1
    )
    cfg.trainer.placement.policy_strict_spread_pg = True
    control = Path(cfg.trainer.export_path)
    control.mkdir(parents=True, exist_ok=True)
    learner_loading = control / LEARNER_LOADING
    inference_started = control / INFERENCE_STARTED
    os.mkfifo(learner_loading)
    os.mkfifo(inference_started)
    # Keep both FIFOs open so select bounds the rendezvous even if a peer fails.
    with learner_loading.open("rb+", buffering=0), inference_started.open("rb+", buffering=0):
        experiment.run_with_exp(cfg, _OverlappingExp)


@pytest.mark.parametrize(
    "run_training",
    [
        pytest.param(_run_barrier_training, id="all-ranks-before-process-group"),
        pytest.param(_run_overlapping_training, id="learner-and-inference-overlap"),
    ],
)
def test_policy_startup(forkserver_context, tmp_path, tiny_policy, run_training):
    run = forkserver_context.Process(target=run_training, args=(tmp_path, tiny_policy))
    run.start()
    run.join(RUN_TIMEOUT_SECONDS)
    if run.exitcode is None:
        run.kill()
        run.join()
        pytest.fail("policy startup did not finish")
    assert run.exitcode == 0
    assert [record["trainer/global_step"] for record in read_metrics(tmp_path) if "policy/raw_grad_norm" in record] == [
        1
    ]
