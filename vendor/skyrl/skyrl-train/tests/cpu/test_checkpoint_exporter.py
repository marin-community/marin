import hashlib
import os
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from huggingface_hub import RepoFile
from safetensors.torch import save_file

from marinskyrl.model_manifest import MODEL_MANIFEST_FILENAME, write_local_model_manifest
from skyrl_train.checkpoint_exporter import CheckpointExportPlan, CheckpointExporter, RayPolicyExportWorkers
from skyrl_train.hf_export_schema import HFExportRequest, HFUploadMode
from skyrl_train.hf_publisher import publish_verified_hf_export
from skyrl_train.hf_model_io import verify_hf_model_export


class FakePolicyExportWorkers:
    def __init__(self):
        self.model_path = None
        self.checkpoint_path = None
        self.export_path = None
        self.tokenizer = None
        self.closed = False

    def initialize(self, model_path: str) -> None:
        self.model_path = model_path

    def load_model_checkpoint(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path

    def save_hf_model(self, export_path: str, tokenizer: object) -> None:
        self.export_path = export_path
        self.tokenizer = tokenizer
        Path(export_path).mkdir(parents=True)
        Path(export_path, "model.safetensors").write_bytes(b"weights")

    def close(self) -> None:
        self.closed = True


class MissingExportWorkers(FakePolicyExportWorkers):
    def save_hf_model(self, export_path: str, tokenizer: object) -> None:
        self.export_path = export_path


class MetadataOnlyExportWorkers(FakePolicyExportWorkers):
    def save_hf_model(self, export_path: str, tokenizer: object) -> None:
        self.export_path = export_path
        Path(export_path).mkdir(parents=True)
        Path(export_path, "config.json").write_text("{}")
        Path(export_path, "model.safetensors.index.json").write_text(
            '{"weight_map": {"model.layers.0.weight": "model-00001-of-00001.safetensors"}}'
        )


class FakeRayActorGroup:
    def __init__(self):
        self.calls = []
        self.killed = False

    def async_run_ray_method(self, dispatch_type, method_name, *args, **kwargs):
        self.calls.append((dispatch_type, method_name, args, kwargs))
        return [f"{method_name}-ref"]

    def kill_actors(self):
        self.killed = True


class FakeHubApi:
    def __init__(self, root, private, corrupt):
        self.root = root
        self.private = private
        self.corrupt = corrupt
        self.commits = {}

    def create_repo(self, *, private, **_kwargs):
        if self.private is None:
            self.private = private

    def model_info(self, _repo_id):
        return SimpleNamespace(private=self.private)

    def upload_folder(self, *, folder_path, path_in_repo, allow_patterns=None, **_kwargs):
        source = Path(folder_path)
        previous = list(self.commits.values())[-1] if self.commits else {}
        files = previous.copy()
        for file in sorted(source.rglob("*")):
            if not file.is_file():
                continue
            relative_path = file.relative_to(source).as_posix()
            if allow_patterns is not None and not any(fnmatchcase(relative_path, pattern) for pattern in allow_patterns):
                continue
            content = file.read_bytes()
            path = (Path(path_in_repo) / relative_path).as_posix()
            files[path] = bytes([content[0] ^ 1]) + content[1:] if path == self.corrupt else content
        commit = f"commit-{len(self.commits)}"
        self.commits[commit] = files
        return SimpleNamespace(oid=commit)

    def get_paths_info(self, _repo_id, paths, *, revision, **_kwargs):
        snapshot = self.commits[revision]
        for path in paths:
            if path in snapshot:
                content = snapshot[path]
                lfs = (
                    {"size": len(content), "oid": hashlib.sha256(content).hexdigest(), "pointerSize": 128}
                    if path.endswith(".safetensors")
                    else None
                )
                yield RepoFile(path=path, size=len(content), oid="blob", lfs=lfs)

    def hf_hub_download(self, _repo_id, filename, *, revision, **_kwargs):
        target = self.root / revision / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.commits[revision][filename])
        return str(target)


def _plan(tmp_path: Path, step: int = 12) -> CheckpointExportPlan:
    checkpoint_path = tmp_path / "checkpoints" / f"global_step_{step}"
    checkpoint_path.mkdir(parents=True)
    (checkpoint_path / "policy").mkdir()
    torch.save({"global_step": step}, checkpoint_path / "trainer_state.pt")
    return CheckpointExportPlan(
        step=step,
        checkpoint_path=str(checkpoint_path),
        export_root=str(tmp_path / "exports"),
        model_path="org/model",
    )


def test_checkpoint_exporter_converts_only_the_policy_model(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    workers = FakePolicyExportWorkers()
    tokenizer = object()

    monkeypatch.setattr("skyrl_train.hf_publisher.HfApi", lambda: pytest.fail("conversion contacted the Hub"))
    result = CheckpointExporter(plan, workers, tokenizer).run()

    assert result.step == 12
    assert result.export_path == str(tmp_path / "exports" / "global_step_12" / "policy")
    assert workers.model_path == "org/model"
    assert workers.checkpoint_path == str(tmp_path / "checkpoints" / "global_step_12" / "policy")
    assert workers.export_path == result.export_path
    assert workers.tokenizer is tokenizer
    assert workers.closed


def test_checkpoint_exporter_rejects_a_mismatched_checkpoint_marker(tmp_path):
    plan = _plan(tmp_path)
    torch.save({"global_step": 11}, Path(plan.checkpoint_path) / "trainer_state.pt")
    workers = FakePolicyExportWorkers()

    with pytest.raises(ValueError, match="checkpoint step mismatch"):
        CheckpointExporter(plan, workers, object()).run()

    assert workers.closed


@pytest.mark.parametrize(
    ("worker_type", "error"),
    [
        (MissingExportWorkers, "no safetensors weights"),
        (MetadataOnlyExportWorkers, "missing 1 referenced safetensors shard"),
    ],
)
def test_checkpoint_exporter_rejects_incomplete_conversion_result(tmp_path, worker_type, error):
    workers = worker_type()
    with pytest.raises(RuntimeError, match=error):
        CheckpointExporter(_plan(tmp_path), workers, object()).run()

    assert workers.closed


def test_unsharded_cloud_export_verification_does_not_head_optional_index(monkeypatch):
    export_path = "s3://bucket/exports/global_step_1/policy"
    monkeypatch.setattr(
        "skyrl_train.hf_model_io.io.list_dir",
        lambda path: [f"{path}/config.json", f"{path}/model.safetensors"],
    )

    def reject_head(path):
        raise AssertionError(f"verification issued an object HEAD request: {path}")

    monkeypatch.setattr("skyrl_train.hf_model_io.io.exists", reject_head)

    verify_hf_model_export(export_path)


def test_ray_policy_export_workers_loads_model_state_without_training_state():
    group = FakeRayActorGroup()
    resolved_refs = []
    workers = RayPolicyExportWorkers(group, resolve=lambda refs: resolved_refs.append(refs))

    workers.initialize("org/model")
    workers.load_model_checkpoint("/checkpoints/global_step_12/policy")
    workers.save_hf_model("/exports/global_step_12/policy", tokenizer="tokenizer")
    workers.close()

    assert group.calls == [
        ("pass_through", "init_model_for_export", ("org/model",), {}),
        (
            "pass_through",
            "load_checkpoint",
            (),
            {
                "ckpt_dir": "/checkpoints/global_step_12/policy",
                "load_training_state": False,
            },
        ),
        (
            "pass_through",
            "save_hf_model",
            ("/exports/global_step_12/policy", "tokenizer"),
            {},
        ),
    ]
    assert resolved_refs == [
        ["init_model_for_export-ref"],
        ["load_checkpoint-ref"],
        ["save_hf_model-ref"],
    ]
    assert group.killed


@pytest.mark.parametrize(
    ("visibility", "requested", "upload_mode", "corrupt", "missing_manifest", "error"),
    [
        (None, None, HFUploadMode.ALL, None, False, None),
        pytest.param(None, None, HFUploadMode.LATEST, None, False, None, id="root-only"),
        (None, True, HFUploadMode.ALL, None, False, None),
        (True, None, HFUploadMode.ALL, None, False, "visibility differs"),
        (False, True, HFUploadMode.ALL, None, False, "visibility differs"),
        (False, None, HFUploadMode.ALL, "config.json", False, "differs from the model manifest"),
        (
            False, None, HFUploadMode.ALL, "checkpoints/step_12/model.safetensors",
            False, "differs from the model manifest",
        ),
        pytest.param(
            None, None, HFUploadMode.LATEST, None, True, "HF export lacks completion markers", id="missing-manifest"
        ),
    ],
)
def test_hf_publisher_checks_visibility_and_uploaded_files(
    tmp_path, monkeypatch, visibility, requested, upload_mode, corrupt, missing_manifest, error
):
    export_path = tmp_path / "exports" / "global_step_12" / "policy"
    export_path.mkdir(parents=True)
    save_file({"weight": torch.tensor([0.0])}, export_path / "model.safetensors")
    for filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (export_path / filename).write_text("{}")
    manifest = write_local_model_manifest(export_path)
    if missing_manifest:
        (export_path / MODEL_MANIFEST_FILENAME).unlink()
    (export_path / "worker.log").write_text("worker log")
    api = FakeHubApi(tmp_path / "downloads", visibility, corrupt)
    monkeypatch.setattr("skyrl_train.hf_publisher.HfApi", lambda: api)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    visibility_request = {} if requested is None else {"hf_hub_private": requested}
    request = HFExportRequest(
        step=12,
        checkpoint_base_path=str(tmp_path / "checkpoints"),
        checkpoint_path=str(tmp_path / "checkpoints" / "global_step_12"),
        export_path=str(tmp_path / "exports"),
        model_path="org/model",
        num_nodes=1,
        gpus_per_node=1,
        hf_hub_repo_id="org/model",
        hf_upload_mode=upload_mode,
        **visibility_request,
    )
    staging_dir = str(tmp_path / "staging")
    if error:
        error_type = RuntimeError if missing_manifest else ValueError
        with pytest.raises(error_type, match=error):
            publish_verified_hf_export(request, staging_dir)
        if corrupt is None:
            assert not api.commits
        if missing_manifest:
            assert api.private is None
        return

    publish_verified_hf_export(request, staging_dir)

    assert api.private is (requested is True)
    paths = [file.path for file in manifest.files] + [MODEL_MANIFEST_FILENAME]
    expected = {path: (export_path / path).read_bytes() for path in paths}
    snapshots = [expected]
    if upload_mode is HFUploadMode.ALL:
        snapshots.append(expected | {f"checkpoints/step_12/{path}": data for path, data in expected.items()})
    assert list(api.commits.values()) == snapshots
    assert os.environ["HF_HUB_OFFLINE"] == "1"
