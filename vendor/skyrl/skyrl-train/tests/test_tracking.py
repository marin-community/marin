"""Tracker cleanup preserves the workload result when final uploads time out."""

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import wandb
from finestore.rl.run_records import RunIndexIdentity
from loguru import logger
from omegaconf import OmegaConf

# Match the CPU suite: Ray workers use the existing locked environment.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
from skyrl_train import trainer as trainer_module
from skyrl_train.entrypoints import main_base
from skyrl_train.entrypoints.main_base import BasePPOExp
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


def test_training_exit_renews_watchdog_for_records_and_timing(tmp_path, monkeypatch):
    events = []
    times = iter((100.0, 125.0))
    monkeypatch.setattr(main_base, "time", SimpleNamespace(monotonic=lambda: next(times)))
    config = OmegaConf.create(
        {
            "trainer": {"export_path": str(tmp_path), "project_name": "test", "run_name": "run", "logger": ["records"]},
            "runtime": {"run_id": "run", "attempt_id": "attempt"},
        }
    )
    trainer = trainer_module.RayPPOTrainer.__new__(trainer_module.RayPPOTrainer)
    trainer.evaluation_wall_time = 7.0

    async def train():
        trainer.start_exit_watchdog()
        raise RuntimeError("training failed")

    async def shutdown():
        events.append("shutdown")

    trainer.train, trainer.shutdown = train, shutdown
    monkeypatch.setattr(trainer, "start_exit_watchdog", lambda: events.append("start"))
    monkeypatch.setattr(trainer, "stop_exit_watchdog", lambda: events.append("stop"))
    experiment = SimpleNamespace(cfg=config, _setup_trainer=lambda _: trainer)
    tracker = BasePPOExp.get_tracker(experiment)
    experiment.get_tracker = lambda: tracker
    store = tracker.logger["records"].store
    close = store.close

    def close_records():
        events.append("close")
        close()

    monkeypatch.setattr(store, "close", close_records)
    monkeypatch.setattr(
        main_base, "write_attempt", lambda _, attempt: events.append(attempt.elapsed_excluding_eval_seconds)
    )
    with pytest.raises(RuntimeError, match="training failed"):
        BasePPOExp._run(experiment)
    assert events == ["start", "shutdown", "start", "close", 18.0, "stop"]
