"""Tracker cleanup preserves the workload result when final uploads time out."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import wandb
from finestore.rl.run_records import RunIndexIdentity
from finestore.rl.run_records import read_attempt
from loguru import logger
from omegaconf import OmegaConf

# Match the CPU suite: Ray workers use the existing locked environment.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
from skyrl_train import trainer as trainer_module
from skyrl_train.entrypoints import main_base
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.utils import tracking as tracking_module
from skyrl_train.utils.tracking import Tracking


@dataclass
class Upload:
    settings: wandb.Settings
    duration: float
    id: str = "shared-run"
    elapsed: float = 0
    uploaded: bool = False
    exit_code: int | None = None
    rows: list[dict] = field(default_factory=list)

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, data, commit):
        self.rows.append(data)

    def finish(self, exit_code=None):
        self.exit_code = exit_code
        deadline = self.settings.finish_timeout or float("inf")
        self.elapsed = min(self.duration, deadline)
        self.uploaded = self.duration <= deadline
        if not self.uploaded and self.settings.finish_timeout_raises:
            raise TimeoutError("final upload timed out")


@pytest.fixture
def local_ray(monkeypatch):
    # Local mode keeps the W&B I/O fake in-process while exercising real Ray actors.
    with ray.init(local_mode=True, num_cpus=1, include_dashboard=False):
        monkeypatch.setattr(
            ray,
            "nodes",
            lambda: [
                {
                    "Alive": True,
                    "NodeManagerAddress": "secondary",
                    "NodeID": ray.get_runtime_context().get_node_id(),
                }
            ],
        )
        yield


@pytest.mark.parametrize("slow_role", [None, "primary", "secondary"])
@pytest.mark.parametrize("training_fails", [False, True])
def test_close_preserves_workload_result_and_reports_upload_timeout(
    local_ray, monkeypatch, tmp_path, slow_role, training_fails
):
    uploads = {}
    wandb_arguments = {}
    warnings = []

    def init(*, settings, **kwargs):
        role = "primary" if settings.x_primary else "secondary"
        wandb_arguments[role] = kwargs
        upload = Upload(settings, duration=120 if role == slow_role else 1, id=kwargs["id"])
        uploads[role] = upload
        return upload

    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(wandb, "init", init)
    sink = logger.add(lambda message: warnings.append(message.record), level="WARNING")
    training_error = RuntimeError("training failed")
    index = RunIndexIdentity(
        run_id="mochi-train-medium-user-1234abcd",
        model="mochi",
        preset="train-medium",
        preset_version="ver",
        user="user",
        branch="topic",
        source_revision="commit",
        dirty=False,
    )

    def workload():
        config = OmegaConf.create(
            {
                "trainer": {"export_path": str(tmp_path)},
                "runtime": {
                    "run_id": "native-run",
                    "attempt_id": "native-attempt",
                    "run_index": index.model_dump(mode="json"),
                },
            }
        )
        with Tracking("native-project", "experiment", backends=("wandb",), config=config) as tracker:
            tracker.log({"loss": 0.5}, step=7)
            if training_fails:
                raise training_error
            return "training completed"

    try:
        if training_fails:
            with pytest.raises(RuntimeError) as error:
                workload()
            assert error.value is training_error
        else:
            assert workload() == "training completed"
    finally:
        logger.remove(sink)

    assert set(uploads) == {"primary", "secondary"}
    assert uploads["primary"].exit_code == int(training_fails)
    assert uploads["primary"].rows == [{"loss": 0.5, "trainer/global_step": 7}]
    for arguments in wandb_arguments.values():
        assert (arguments["project"], arguments["id"]) == ("marin-rl", "native-attempt")
        assert arguments["name"] == arguments["group"] == "mochi-train-medium-user-1234abcd"
    assert wandb_arguments["primary"]["tags"] == [
        "model:mochi",
        "preset:train-medium",
        "version:ver",
        "user:user",
        "branch:topic",
    ]
    for role, upload in uploads.items():
        assert upload.uploaded == (role != slow_role)
        assert upload.elapsed < 120
    assert len(warnings) == int(slow_role is not None)


@pytest.mark.parametrize("training_fails", [False, True])
def test_training_exit_uses_separate_deadlines_for_trainer_and_records(tmp_path, monkeypatch, training_fails):
    times = iter((100.0, 125.0))
    monkeypatch.setattr(main_base, "time", SimpleNamespace(monotonic=lambda: next(times)))
    failure = RuntimeError("training failed")
    timers, exits, executor_shutdown = [], [], []
    close_exits, timing_exits = [], []

    class Timer:
        def __init__(self, _timeout, callback):
            self.callback = callback
            self.armed = False
            timers.append(self)

        def start(self):
            self.armed = True

        def cancel(self):
            self.armed = False

        def join(self):
            pass

        def expire(self):
            if self.armed:
                self.callback()

    clock = SimpleNamespace(Timer=Timer)
    monkeypatch.setattr(trainer_module, "threading", clock)
    monkeypatch.setattr(trainer_module, "os", SimpleNamespace(_exit=exits.append))
    monkeypatch.setattr(tracking_module, "threading", clock)
    monkeypatch.setattr(tracking_module, "os", SimpleNamespace(_exit=exits.append))

    trainer = trainer_module.RayPPOTrainer.__new__(trainer_module.RayPPOTrainer)
    trainer.evaluation_wall_time = 7.0

    class RecordingExecutor(ThreadPoolExecutor):
        def shutdown(self, *args, **kwargs):
            executor_shutdown.append(timers[0].armed)
            super().shutdown(*args, **kwargs)

    async def train():
        trainer._start_exit_watchdog()
        asyncio.get_running_loop().set_default_executor(RecordingExecutor())
        if training_fails:
            raise failure

    async def shutdown():
        asyncio.get_running_loop().set_default_executor(RecordingExecutor())

    def expire_active_timers():
        for timer in timers:
            timer.expire()
        return tuple(exits)

    write_attempt = main_base.write_attempt

    def write_timing(*args):
        timing_exits.append(expire_active_timers())
        write_attempt(*args)

    monkeypatch.setattr(main_base, "write_attempt", write_timing)

    trainer.train = train
    trainer.shutdown = shutdown
    config = OmegaConf.create(
        {
            "trainer": {
                "export_path": str(tmp_path),
                "project_name": "test",
                "run_name": "native-run",
                "logger": ["records"],
            },
            "runtime": {"run_id": "native-run", "attempt_id": "native-attempt"},
        }
    )
    experiment = SimpleNamespace(cfg=config, _setup_trainer=lambda tracker: trainer)
    tracker = BasePPOExp.get_tracker(experiment)
    store = tracker.logger["records"].store
    close = store.close

    def close_records():
        close_exits.append(expire_active_timers())
        close()

    monkeypatch.setattr(store, "close", close_records)
    experiment.get_tracker = lambda: tracker
    if training_fails:
        with pytest.raises(RuntimeError) as raised:
            BasePPOExp._run(experiment)
        assert raised.value is failure
    else:
        BasePPOExp._run(experiment)
    assert executor_shutdown and all(executor_shutdown)
    assert close_exits == [(1,)]
    assert timing_exits == [(1, 1)]
    assert expire_active_timers() == (1, 1)
    assert (attempt := read_attempt(str(tmp_path), "native-attempt")) is not None
    assert attempt.elapsed_excluding_eval_seconds == 18.0
