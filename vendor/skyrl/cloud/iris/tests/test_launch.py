"""Behavior tests for the single-config launch lifecycle."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

import numpy as np
import pytest
import yaml
from finestore.rl.run_records import read_attempt
from rigging.filesystem.storage_path import StoragePath
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.types import JobName
from omegaconf import OmegaConf
from safetensors.numpy import save

from skyrl_train.hf_export_schema import ExportFailed

from cloud.iris import export_hf_checkpoint, hf_model_cache, launch, task_runtime
from cloud.iris.artifacts import CHECKPOINT_MARKER_FILENAME
from cloud.iris.task_runtime import prepare_draft_model
from marinskyrl.speculative_decoding import SpeculatorModelConfig
from cloud.iris.hf_model_cache import HuggingFaceSnapshot, load_model_manifest
from cloud.iris.launch import LaunchOutcome, LaunchState, execute_launch
from cloud.iris.launch_config import load_launch_config
from cloud.iris.tests.launch_config_fixtures import raw_launch_config
from cloud.iris.tests.model_snapshot_fixtures import memory_model_snapshot
from cloud.iris.iris_backend import supervise_iris_job
from iris.resources.state import JobState
from rigging.timing import Timestamp
from marinskyrl.checkpoint_paths import BEST_CHECKPOINT_FILENAME, HF_EXPORT_REQUEST_FILENAME, policy_export_path
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.checkpoint_listing import ROLLOUT_STATE_FILENAME
from skyrl_train.hf_export import write_hf_export_request
from skyrl_train.hf_export_schema import HFExportRequest, TRAINER_STATE_FILENAME


@dataclass
class RecordingBackend:
    outcome: LaunchOutcome = LaunchOutcome("/user/job", "succeeded", 0)
    validated: bool = False
    launched: bool = False
    launched_config: dict | None = None
    exported: bool = False

    def validate(self, _config_path: Path) -> None:
        self.validated = True

    def launch(self, config_path: Path) -> LaunchOutcome:
        self.launched = True
        self.launched_config = yaml.safe_load(config_path.read_text())
        return self.outcome

    def export_terminal_policy(self, _config_path: Path) -> None:
        self.exported = True


def _config_path(tmp_path: Path, *, submission: str = "wait") -> Path:
    config = raw_launch_config()
    config["run"]["submission"] = submission
    config["artifacts"].update(
        checkpoint_root=str(tmp_path / "checkpoints"),
        export_root=str(tmp_path / "exports"),
        attempts_root=str(tmp_path / "attempts"),
        resolved_config_uri=str(tmp_path / "resolved.yaml"),
        terminal_manifest_uri=str(tmp_path / "terminal.json"),
    )
    config["inputs"]["train_data"] = [
        {
            "uri": "s3://data/gsm8k",
            "identity": "sha256:gsm8k",
            "local_path": "/tmp/data/gsm8k",
            "relative_path": "train.parquet",
            "kind": "directory",
        }
    ]
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_prepare_validates_without_submitting(tmp_path: Path) -> None:
    backend = RecordingBackend()

    result = execute_launch(_config_path(tmp_path, submission="prepare"), backend=backend)

    assert result.state is LaunchState.PREPARED
    assert backend.validated
    assert not backend.launched


@pytest.mark.parametrize(("previous_step", "next_step"), [(None, 2), (2, None), (2, 5), (2, 2)])
def test_existing_run_rejects_changed_starting_checkpoint(tmp_path, previous_step, next_step):
    path = _config_path(tmp_path, submission="prepare")
    config = yaml.safe_load(path.read_text())
    config["artifacts"] = {key: str(tmp_path / key) for key in config["artifacts"]}
    previous_inputs = {}
    for step, inputs in ((previous_step, previous_inputs), (next_step, config["inputs"])):
        if step is not None:
            inputs["starting_checkpoint"] = {"uri": f"s3://seed/global_step_{step}", "identity": f"seed@{step}"}
    path.write_text(yaml.safe_dump(config))
    resolved = Path(config["artifacts"]["resolved_config_uri"])
    snapshot = json.dumps({"config": {"inputs": previous_inputs}})
    resolved.write_text(snapshot)
    backend = RecordingBackend()

    if previous_step == next_step:
        assert execute_launch(path, backend=backend).state is LaunchState.PREPARED
    else:
        with pytest.raises(ValueError, match="new RL artifact version"):
            execute_launch(path, backend=backend)
    assert resolved.read_text() == snapshot


def test_detach_submits_without_exporting(tmp_path: Path) -> None:
    backend = RecordingBackend()

    result = execute_launch(_config_path(tmp_path, submission="detach"), backend=backend)

    assert result.state is LaunchState.SUBMITTED
    assert backend.launched
    assert not backend.exported


def test_wait_does_not_commit_nonterminal_zero_exit_outcome(tmp_path: Path) -> None:
    backend = RecordingBackend(outcome=LaunchOutcome("/user/still-running", "submitted", 0))

    result = execute_launch(_config_path(tmp_path), backend=backend)

    assert result.state is LaunchState.FAILED
    assert result.failure == "Iris job reached submitted"
    assert not backend.exported
    record = read_attempt(str(tmp_path / "exports"), "attempt-1")
    assert record is not None
    assert record.native.result.model_dump(mode="json") == result.model_dump(mode="json")
    assert not StoragePath(str(tmp_path / "terminal.json")).exists()


@dataclass
class LocalLaunchBackend:
    """Iris I/O boundary with local resolved-config and export fixtures."""

    export_jobs: list[str] = field(default_factory=list)
    outcome: LaunchOutcome = LaunchOutcome("/user/training", "succeeded", 0)
    checkpoint_marker: str | None = "3\n"
    export_step: int = 3
    export_fails: bool = False

    def validate(self, _config_path: Path) -> None:
        raise AssertionError("this lifecycle test submits training")

    def launch(self, config_path: Path) -> LaunchOutcome:
        config = yaml.safe_load(config_path.read_text())
        artifacts = config["artifacts"]
        resolved = Path(artifacts["resolved_config_uri"])
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(config_path.read_bytes())
        checkpoint_root = Path(artifacts["checkpoint_root"])
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_marker is not None:
            (checkpoint_root / CHECKPOINT_MARKER_FILENAME).write_text(self.checkpoint_marker)
        status = SimpleNamespace(
            state=JobState(self.outcome.job_state),
            started_at=Timestamp.from_seconds(datetime(2026, 11, 7, 1, tzinfo=timezone.utc).timestamp()),
            finished_at=Timestamp.from_seconds(datetime(2026, 11, 7, 2, tzinfo=timezone.utc).timestamp()),
        )
        return supervise_iris_job(SimpleNamespace(job_id=self.outcome.job_id, wait=lambda **_kwargs: status))

    def export_terminal_policy(self, config_path: Path) -> None:
        if self.export_fails:
            raise ExportFailed("Hub job did not finish")
        config = yaml.safe_load(config_path.read_text())
        checkpoint_root = config["artifacts"]["checkpoint_root"]
        assert (Path(checkpoint_root) / CHECKPOINT_MARKER_FILENAME).read_text().strip() == "3"
        output = policy_export_path(config["artifacts"]["export_root"], self.export_step)
        self.export_jobs.append(output)
        root = Path(output)
        root.mkdir(parents=True)
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            (root / name).write_text("{}")


@pytest.mark.parametrize(
    "export_hf,outcome,own_tokenizer,export_fails",
    [
        (None, LaunchOutcome("/user/training", "succeeded", 0), False, False),
        (True, LaunchOutcome("/user/training", "succeeded", 0), False, False),
        (True, LaunchOutcome("/user/training", "succeeded", 0), True, False),
        (None, LaunchOutcome("/user/training", "failed", 1), False, False),
        (True, LaunchOutcome("/user/training", "succeeded", 0), False, True),
    ],
    ids=["default-native", "explicit-hf", "embedded-tokenizer-hf", "failed-native", "failed-hub"],
)
def test_launch_persists_attempt_result_and_exports_only_on_opt_in(
    tmp_path: Path, export_hf: bool | None, outcome, own_tokenizer: bool, export_fails: bool
):
    path = _config_path(tmp_path)
    config = yaml.safe_load(path.read_text())
    if export_hf is None:
        config["run"].pop("export_hf")
    else:
        config["run"]["export_hf"] = export_hf
    if own_tokenizer:
        config["inputs"]["model"].update(tokenizer_uri=None, tokenizer_revision=None)
    config["skyrl"]["trainer"].update(max_steps=3, eval_interval=3, ckpt_interval=3)
    path.write_text(yaml.safe_dump(config))
    backend = LocalLaunchBackend(outcome=outcome, export_fails=export_fails)

    result = execute_launch(path, backend=backend)
    saved = read_attempt(str(tmp_path / "exports"), "attempt-1")
    assert saved is not None
    assert saved.iris_job_id == outcome.job_id
    attempt = saved.native.model_dump(mode="json")
    assert saved.native.result.started_at == "2026-11-07T01:00:00+00:00"
    assert saved.native.result.finished_at == "2026-11-07T02:00:00+00:00"
    assert not (tmp_path / "attempts").exists()
    if outcome.exit_code:
        assert result.state is LaunchState.FAILED
        assert attempt["result"]["state"] == "failed"
        assert attempt["result"]["failure"] == "Iris job reached failed"
        assert not (tmp_path / "terminal.json").exists()
        assert backend.export_jobs == []
        return
    if export_fails:
        assert result.state is LaunchState.FAILED and attempt["result"]["state"] == "failed"
        assert attempt["result"]["failure"] == "Terminal policy export failed: Hub job did not finish"
        assert not (tmp_path / "terminal.json").exists()
        return
    assert result.state is LaunchState.SUCCEEDED
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal == attempt
    assert terminal["config"] == yaml.safe_load((tmp_path / "resolved.yaml").read_text())
    assert terminal["result"]["state"] == "succeeded"
    if export_hf:
        assert backend.export_jobs == [str(tmp_path / "exports/global_step_3/policy")]
        assert result.model is not None and result.model.global_step == 3
        assert terminal["result"]["model"]["policy_export_uri"] == backend.export_jobs[0]
        for key in ("tokenizer_uri", "tokenizer_revision"):
            assert terminal["result"]["model"][key] == config["inputs"]["model"][key]
    else:
        assert backend.export_jobs == []
        assert result.model is None and terminal["result"]["model"] is None
        assert not (tmp_path / "exports/global_step_3").exists()


@pytest.fixture
def completed_run_checkpoints(tmp_path):
    path = _config_path(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["run"]["export_hf"] = False
    config["skyrl"]["trainer"].update(max_steps=3, eval_interval=3, ckpt_interval=3, checkpoint_retention="best")
    path.write_text(yaml.safe_dump(config))
    root = tmp_path / "checkpoints"
    for step in (1, 2, 3, 4):
        checkpoint = root / f"global_step_{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / TRAINER_STATE_FILENAME).touch()
        (checkpoint / ROLLOUT_STATE_FILENAME).touch()
    return path, root


@dataclass(frozen=True)
class RetentionCase:
    kept: tuple[int, ...]
    retention: str | None = "best"
    keep_every: int | None = None
    best_format: str | None = "native"
    export_hf: bool = False
    job_state: str = "succeeded"
    missing_rollout_step: int | None = None
    marker: str | None = "3"
    expected_marker: str | None = "3"
    pending_exports: tuple[int, ...] = ()
    inventory_error: bool = False


@pytest.mark.parametrize("case", [
    pytest.param(RetentionCase((1,), export_hf=True, expected_marker="1"), id="best-after-export"),
    pytest.param(RetentionCase((1, 2), keep_every=2, expected_marker="2"), id="periodic-newer-than-best"),
    pytest.param(RetentionCase((1, 3), keep_every=1, missing_rollout_step=2), id="missing-rollout"),
    pytest.param(RetentionCase((2,), keep_every=2, best_format="hf", expected_marker="2"), id="hf-with-periodic"),
    pytest.param(RetentionCase((3,), best_format=None), id="no-winner"),
    pytest.param(RetentionCase((2, 3), retention="latest", keep_every=2, best_format=None), id="latest"),
    pytest.param(RetentionCase((1, 2, 3, 4), retention=None), id="legacy-config"),
    pytest.param(RetentionCase((1, 2, 3, 4), job_state="failed"), id="failed"),
    pytest.param(RetentionCase((1, 2, 3, 4), best_format=None, marker=None, expected_marker=None), id="no-marker"),
    pytest.param(RetentionCase((1, 2, 3, 4), best_format=None, marker="invalid", expected_marker="invalid"), id="unreadable-marker"),
    pytest.param(RetentionCase((1, 2, 4), pending_exports=(2, 4), expected_marker="2"), id="pending-exports"),
    pytest.param(
        RetentionCase((1, 2, 3, 4), keep_every=2, best_format="hf", inventory_error=True),
        id="checkpoint-storage-error",
    ),
])
def test_launch_prunes_checkpoints_after_the_terminal_receipt(completed_run_checkpoints, case, monkeypatch):
    path, root = completed_run_checkpoints
    config = yaml.safe_load(path.read_text())
    config["run"]["export_hf"] = case.export_hf
    trainer = config["skyrl"]["trainer"]
    if case.retention is None:
        trainer.pop("checkpoint_retention")
    else:
        trainer.update(checkpoint_retention=case.retention, checkpoint_keep_every=case.keep_every)
    if case.export_hf:
        trainer["hf_export_checkpoint"] = "best"
    path.write_text(yaml.safe_dump(config))

    if case.missing_rollout_step is not None:
        (root / f"global_step_{case.missing_rollout_step}" / ROLLOUT_STATE_FILENAME).unlink()
    if case.best_format is not None:
        (root / BEST_CHECKPOINT_FILENAME).write_text(
            json.dumps({"step": 1 if case.best_format == "native" else 0, "model_format": case.best_format})
        )
    if case.pending_exports:
        training = load_launch_config(path)
        for step in case.pending_exports:
            write_hf_export_request(HFExportRequest.from_trainer(
                training.skyrl.trainer, step=step, checkpoint_root=str(root), export_root=training.artifacts.export_root
            ))
    if case.inventory_error:
        listing = launch.checkpoint_io.list_dir

        def fail_first_listing(checkpoint_path):
            if checkpoint_path == str(root):
                monkeypatch.setattr(launch.checkpoint_io, "list_dir", listing)
                raise OSError("checkpoint storage unavailable")
            return listing(checkpoint_path)

        monkeypatch.setattr(launch.checkpoint_io, "list_dir", fail_first_listing)

    backend = LocalLaunchBackend(
        outcome=LaunchOutcome("/user/training", case.job_state, int(case.job_state == "failed")),
        export_step=1,
        checkpoint_marker=case.marker,
    )
    result = execute_launch(path, backend=backend)

    assert result.state.value == case.job_state
    assert {path.name for path in root.glob("global_step_*")} == {f"global_step_{step}" for step in case.kept}
    if case.job_state == "succeeded":
        terminal = Path(config["artifacts"]["terminal_manifest_uri"])
        assert json.loads(terminal.read_text())["result"]["state"] == "succeeded"
    marker = root / CHECKPOINT_MARKER_FILENAME
    assert (marker.read_text().strip() if marker.exists() else None) == case.expected_marker
    if case.export_hf:
        assert result.model is not None and result.model.global_step == 1


def test_checkpoint_export_launch_preserves_training_checkpoints(completed_run_checkpoints, tmp_path):
    path, root = completed_run_checkpoints
    training = load_launch_config(path)
    request = HFExportRequest.from_trainer(
        training.skyrl.trainer, step=1, checkpoint_root=str(root), export_root=training.artifacts.export_root
    )
    spec = export_hf_checkpoint.ExportJobSpec(
        request=request, cluster="cw-rno2a", priority="batch", gpu_variant="H100", job_name="export", timeout=7200,
        no_wait=False,
    )
    export_path = tmp_path / "export.yaml"
    OmegaConf.save(export_hf_checkpoint.checkpoint_export_launch_config(training, request, spec), export_path)
    (root / CHECKPOINT_MARKER_FILENAME).write_text("3\n")
    before = set(root.glob("global_step_*"))

    result = execute_launch(export_path, backend=LocalLaunchBackend(checkpoint_marker=None))

    assert result.state is LaunchState.SUCCEEDED
    assert set(root.glob("global_step_*")) == before
    assert (root / CHECKPOINT_MARKER_FILENAME).read_text().strip() == "3"


def test_checkpoint_pruning_waits_for_the_terminal_receipt(completed_run_checkpoints, monkeypatch):
    path, root = completed_run_checkpoints
    before = set(root.glob("global_step_*"))

    def fail_terminal_write(_uri, _payload):
        raise OSError("terminal receipt storage is unavailable")

    monkeypatch.setattr(launch, "write_json", fail_terminal_write)
    with pytest.raises(OSError, match="terminal receipt storage is unavailable"):
        execute_launch(path, backend=LocalLaunchBackend())

    assert set(root.glob("global_step_*")) == before
    assert (root / CHECKPOINT_MARKER_FILENAME).read_text().strip() == "3"


@pytest.mark.parametrize("initializer_wins", [True, False], ids=["step-zero", "none-eligible"])
def test_launch_publishes_the_selected_best_policy(completed_run_checkpoints, tmp_path: Path, initializer_wins: bool):
    path, checkpoint_root = completed_run_checkpoints
    config = yaml.safe_load(path.read_text())
    initializer_uri = config["inputs"]["model"]["uri"]
    config["run"]["export_hf"] = True
    config["skyrl"]["trainer"]["hf_export_checkpoint"] = "best"
    path.write_text(yaml.safe_dump(config))

    class BestCheckpointBackend(LocalLaunchBackend):
        def launch(self, config_path: Path) -> LaunchOutcome:
            outcome = super().launch(config_path)
            if initializer_wins:
                (checkpoint_root / BEST_CHECKPOINT_FILENAME).write_text(
                    json.dumps({"step": 0, "model_format": "hf", "model_uri": initializer_uri})
                )
            trainer = RayPPOTrainer.__new__(RayPPOTrainer)
            trainer.cfg = OmegaConf.load(config_path).skyrl
            trainer.global_step = 3
            trainer.all_timings = {}
            trainer.handle_hf_export()
            assert not list(checkpoint_root.glob(f"global_step_*/{HF_EXPORT_REQUEST_FILENAME}"))
            return outcome

        def export_terminal_policy(self, config_path: Path) -> None:
            export_hf_checkpoint.export_terminal_policy(config_path)

    result = execute_launch(path, backend=BestCheckpointBackend())

    assert result.state is LaunchState.SUCCEEDED, result.failure
    terminal_model = json.loads((tmp_path / "terminal.json").read_text())["result"]["model"]
    if initializer_wins:
        assert result.model is not None and result.model.global_step == 0
        assert (terminal_model["global_step"], terminal_model["policy_export_uri"]) == (0, initializer_uri)
        assert terminal_model["tokenizer_uri"] == config["inputs"]["model"]["tokenizer_uri"]
        assert not list(checkpoint_root.glob("global_step_*"))
        assert not (checkpoint_root / CHECKPOINT_MARKER_FILENAME).exists()
    else:
        assert terminal_model is result.model is None
    assert not list(checkpoint_root.glob(f"global_step_*/{HF_EXPORT_REQUEST_FILENAME}"))
    assert not list((tmp_path / "exports").glob("global_step_*"))


def _hub_config_path(tmp_path: Path, *, submission: str = "detach") -> Path:
    path = _config_path(tmp_path, submission=submission)
    config = yaml.safe_load(path.read_text())
    config["inputs"]["model"].update(
        uri="org/policy",
        identity="a" * 40,
        local_path="org/policy",
        tokenizer_uri="org/policy",
        tokenizer_revision="a" * 40,
    )
    config["iris"]["allocation"]["num_nodes"] = 2
    skyrl = config["skyrl"]
    skyrl["trainer"]["algorithm"].update(
        advantage_estimator="uniform",
        distillation={
            "objective": "sampled_reverse_kl",
            "routing_plan": "opd",
            "coefficient": 1.0,
            "reward_mode": "replace",
        },
    )
    skyrl["teachers"] = {
        "local": {
            "source": "local_inference",
            "placement": "pinned",
            "backend": "vllm",
            "evidence": "chosen_token",
            "model": {"path": "org/teacher", "revision": "b" * 40},
            "resources": {"num_nodes": 1, "gpus_per_node": 8, "tensor_parallel_size": 1, "colocation_group": "teacher"},
        },
        "external": {
            "source": "openai_compatible",
            "placement": "external",
            "evidence": "chosen_token",
            "model": {"path": "org/external", "revision": "c" * 40},
            "endpoints": [{"url": "https://teacher.example/v1", "max_concurrency": 8}],
            "tokenizer_fingerprint": "sha256:" + "d" * 64,
            "max_sequence_length": 1024,
            "request_timeout_seconds": 120,
        },
    }
    skyrl["teacher_routing"] = {
        "opd": {"revision": "v1", "routes": {name: {"teacher": name, "weight": 1.0} for name in ("local", "external")}}
    }
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("separate_tokenizer", [False, True])
@pytest.mark.parametrize(("hub_failure", "shared_model"), [(False, False), (False, True), (True, False)])
def test_cold_policy_and_local_teachers_are_mirrored_before_gpu_submission(
    tmp_path, monkeypatch, hub_failure, shared_model, separate_tokenizer
):
    path = _hub_config_path(tmp_path)
    if separate_tokenizer:
        config = yaml.safe_load(path.read_text())
        config["inputs"]["model"]["tokenizer_uri"] = "org/tokenizer"
        path.write_text(yaml.safe_dump(config))
    if shared_model:
        config = yaml.safe_load(path.read_text())
        config["skyrl"]["teachers"]["local"]["model"] = {"path": "org/policy", "revision": "a" * 40}
        path.write_text(yaml.safe_dump(config))
    expected_models = ["org/policy"] if shared_model else ["org/policy", "org/teacher"]
    snapshot = memory_model_snapshot(f"{tmp_path.name}/hub")
    mirror_root = tmp_path / "mirrors"
    monkeypatch.setattr(
        hf_model_cache, "marin_temp_bucket", lambda _ttl, *, prefix, source_prefix: str(mirror_root / prefix)
    )
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)

    def open_snapshot(model_id, _revision):
        if hub_failure and model_id == "org/teacher":
            raise OSError("Hub unavailable")
        if separate_tokenizer and not shared_model and model_id == "org/policy":
            return HuggingFaceSnapshot(
                snapshot.filesystem,
                snapshot.root,
                tuple(entry for entry in snapshot.files if not entry.path.startswith("tokenizer")),
            )
        return snapshot

    monkeypatch.setattr(hf_model_cache, "_open_hugging_face_snapshot", open_snapshot)

    class MirroredBackend(RecordingBackend):
        def launch(self, config_path):
            config = yaml.safe_load(config_path.read_text())
            mirrors = config["inputs"]["hugging_face_models"]
            assert {model["model_id"] for model in mirrors} == set(expected_models)
            for model in mirrors:
                manifest = load_model_manifest(model["uri"])
                assert manifest.identity == model["identity"]
                assert manifest.revision == model["resolved_revision"]
                assert Path(model["uri"], "model.safetensors").read_bytes() == save(
                    {"weight": np.arange(4, dtype=np.float32)}
                )
            return super().launch(config_path)

    backend = MirroredBackend()
    previous = get_job_info()
    set_job_info(JobInfo(task_id=JobName.from_wire("/user/coordinator/0")))
    try:
        if hub_failure:
            with pytest.raises(OSError, match="Hub unavailable"):
                execute_launch(path, backend=backend)
            assert not backend.launched
        else:
            result = execute_launch(path, backend=backend)
            assert result.state is LaunchState.SUBMITTED
    finally:
        set_job_info(previous)


@pytest.mark.parametrize("submission", ["prepare", "detach"])
def test_hub_models_are_not_downloaded_on_a_local_launcher(tmp_path, monkeypatch, submission):
    backend = RecordingBackend()
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)
    path = _hub_config_path(tmp_path, submission=submission)
    previous = get_job_info()
    set_job_info(None)
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    try:
        if submission == "prepare":
            result = execute_launch(path, backend=backend)
            assert result.state is LaunchState.PREPARED
        else:
            with pytest.raises(RuntimeError, match="Iris CPU coordinator"):
                execute_launch(path, backend=backend)
        assert not backend.launched
    finally:
        set_job_info(previous)


@pytest.mark.parametrize("tokenizer_uri", ["org/policy", "", None])
def test_policy_without_a_separate_tokenizer_rejects_model_only_mirrors(tmp_path, monkeypatch, tokenizer_uri):
    path = _hub_config_path(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["inputs"]["model"]["tokenizer_uri"] = tokenizer_uri
    path.write_text(yaml.safe_dump(config))
    snapshot = memory_model_snapshot(f"{tmp_path.name}/hub")
    model_only = HuggingFaceSnapshot(
        snapshot.filesystem,
        snapshot.root,
        tuple(entry for entry in snapshot.files if not entry.path.startswith("tokenizer")),
    )
    monkeypatch.setattr(hf_model_cache, "_open_hugging_face_snapshot", lambda *_: model_only)
    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_, **__: str(tmp_path / "mirror"))
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)
    backend = RecordingBackend()
    previous = get_job_info()
    set_job_info(JobInfo(task_id=JobName.from_wire("/user/coordinator/0")))
    try:
        with pytest.raises(ValueError, match="tokenizer"):
            execute_launch(path, backend=backend)
        assert not backend.launched
    finally:
        set_job_info(previous)


def test_hub_draft_is_mirrored_and_rewritten_before_gpu_submission(tmp_path, monkeypatch):
    path = _config_path(tmp_path, submission="detach")
    config = yaml.safe_load(path.read_text())
    config["skyrl"]["trainer"]["placement"]["colocate_all"] = False
    config["iris"]["allocation"]["num_nodes"] = 2
    config["skyrl"]["generator"]["speculative_decoding"] = {
        "model": {"source_uri": "hf://org/draft", "source_identity": "d" * 40},
        "num_speculative_tokens": 3,
        "method": "eagle3",
    }
    path.write_text(yaml.safe_dump(config))
    snapshot = memory_model_snapshot(f"{tmp_path.name}/hub")
    snapshot = HuggingFaceSnapshot(
        snapshot.filesystem,
        snapshot.root,
        tuple(entry for entry in snapshot.files if not entry.path.startswith("tokenizer")),
    )
    opened = []

    def open_snapshot(model_id, revision):
        opened.append((model_id, revision))
        return snapshot

    monkeypatch.setattr(hf_model_cache, "_open_hugging_face_snapshot", open_snapshot)
    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_, **__: str(tmp_path / "draft"))
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)

    class DraftBackend(RecordingBackend):
        def launch(self, config_path):
            resolved = yaml.safe_load(config_path.read_text())
            draft = SpeculatorModelConfig.from_mapping(
                resolved["skyrl"]["generator"]["speculative_decoding"]["model"], context="draft"
            )
            manifest = load_model_manifest(draft.source_uri)
            assert manifest.tokenizer_mode == "policy"
            assert draft.source_identity == manifest.identity
            assert Path(draft.source_uri, "model.safetensors").is_file()
            assert not Path(draft.source_uri, "tokenizer.json").exists()
            assert resolved["inputs"]["hugging_face_models"][0]["uri"] == draft.source_uri
            assert prepare_draft_model(draft, cache_ttl_days=14, cache_source_prefix="s3://region/run") == draft
            return super().launch(config_path)

    previous = get_job_info()
    set_job_info(JobInfo(task_id=JobName.from_wire("/user/coordinator/0")))
    try:
        result = execute_launch(path, backend=DraftBackend())
        assert result.state is LaunchState.SUBMITTED
    finally:
        set_job_info(previous)
    assert opened == [("org/draft", "d" * 40)]


def test_unsupported_policy_layout_retains_node_fallback(tmp_path, monkeypatch):
    path = _hub_config_path(tmp_path)
    snapshot = memory_model_snapshot(f"{tmp_path.name}/hub")
    unsupported = HuggingFaceSnapshot(
        snapshot.filesystem,
        snapshot.root,
        tuple(entry for entry in snapshot.files if not entry.path.endswith(".safetensors")),
    )
    monkeypatch.setattr(
        hf_model_cache,
        "_open_hugging_face_snapshot",
        lambda model_id, _: unsupported if model_id == "org/policy" else snapshot,
    )
    monkeypatch.setattr(
        hf_model_cache, "marin_temp_bucket", lambda _ttl, *, prefix, source_prefix: str(tmp_path / "mirrors" / prefix)
    )
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)
    backend = RecordingBackend()
    previous = get_job_info()
    set_job_info(JobInfo(task_id=JobName.from_wire("/user/coordinator/0")))
    try:
        result = execute_launch(path, backend=backend)
        assert result.state is LaunchState.SUBMITTED
    finally:
        set_job_info(previous)
    assert [model["model_id"] for model in backend.launched_config["inputs"]["hugging_face_models"]] == ["org/teacher"]
    downloaded = []
    node_snapshot = tmp_path / "fallback"
    node_snapshot.mkdir()

    def download(model_id, *, revision, allow_patterns):
        downloaded.append((model_id, revision))
        return node_snapshot

    monkeypatch.setattr(task_runtime, "download_hugging_face_snapshot", download)
    monkeypatch.setattr(task_runtime.huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "node-cache"))
    task_runtime.stage_hugging_face_models(OmegaConf.create(backend.launched_config))
    assert downloaded == [("org/policy", "a" * 40)]
