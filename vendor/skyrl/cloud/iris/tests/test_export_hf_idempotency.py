"""Exercise conversion, CPU publication, and retry at the Iris/Hub boundaries."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import subprocess
from types import SimpleNamespace

import cloudpickle
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem
import numpy as np
import pytest
from safetensors.numpy import save_file

from cloud.iris import export_hf_checkpoint
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp, region_constraint
from iris.resources.state import JobState
from marinskyrl.checkpoint_paths import policy_export_path
from marinskyrl.model_manifest import write_local_model_manifest
from skyrl_train.hf_export import read_hf_export_request
from skyrl_train.hf_export_schema import HFExportRequest, HFExportStatus, HFUploadMode
from skyrl_train.io import io


class ObjectStore(MemoryFileSystem):
    @classmethod
    def _strip_protocol(cls, path):
        return super()._strip_protocol(str(path).removeprefix("s3://").removeprefix("gs://"))


@pytest.fixture
def export_run(tmp_path, monkeypatch):
    store = ObjectStore()
    store.store.clear()
    monkeypatch.setattr(
        io, "filesystem_and_path",
        lambda path: (store, path) if path.startswith(("s3://", "gs://")) else (LocalFileSystem(), path),
    )
    monkeypatch.setattr(export_hf_checkpoint, "get_job_info", lambda: None)
    events, submissions, uploads, failures = [], [], [], []
    request = HFExportRequest(
        step=10,
        checkpoint_base_path=str(tmp_path / "checkpoints"),
        checkpoint_path=str(tmp_path / "checkpoints/global_step_10"),
        export_path="s3://fixture/exports",
        model_path="org/model",
        num_nodes=4,
        gpus_per_node=8,
        hf_hub_repo_id="org/published",
        hf_hub_private=True,
        hf_hub_revision="release",
    )
    spec = export_hf_checkpoint.ExportJobSpec(
        request=request, cluster="cw-rno2a", target_cluster="cw-rno2a",
        parent_cluster_config="lib/iris/config/marin.yaml", gpu_variant="H100",
        priority="batch", job_name="export-test", timeout=60, no_wait=False,
    )

    def save_export():
        local = tmp_path / "model"
        local.mkdir(exist_ok=True)
        (local / "config.json").write_text("{}")
        (local / "tokenizer.json").write_text("{}")
        save_file({"weight": np.arange(4, dtype=np.float32)}, local / "model.safetensors")
        write_local_model_manifest(local)
        target = policy_export_path(request.export_path, request.step)
        for file in local.iterdir():
            with store.open(f"{target}/{file.name}", "wb") as destination:
                destination.write(file.read_bytes())

    def convert(_command, **_kwargs):
        events.append("gpu")
        save_export()
        events.append("gpu-finished")
        return 0

    monkeypatch.setattr(subprocess, "call", convert)

    class Hub:
        def create_repo(self, **kwargs):
            assert kwargs["private"] is True

        def upload_folder(self, **kwargs):
            if failures:
                failures.pop()
                raise RuntimeError("Hub unavailable")
            assert kwargs["revision"] == "release"
            assert (Path(kwargs["folder_path"]) / "model.safetensors").is_file()
            uploads.append(kwargs["path_in_repo"])

    monkeypatch.setattr("skyrl_train.hf_publisher.HfApi", Hub)

    class Job:
        job_id = "/test/cpu-hub"

        def wait(self, **_kwargs):
            events.append("cpu")
            function, args, kwargs = cloudpickle.loads(submissions[-1]["entrypoint"].workdir_files["_callable.pkl"])
            try:
                function(*args, **kwargs)
            except RuntimeError:
                return SimpleNamespace(state=JobState.FAILED)
            return SimpleNamespace(state=JobState.SUCCEEDED)

        def cancel(self):
            events.append("cancel")

    @contextmanager
    def connect(*_args, **_kwargs):
        def submit(**kwargs):
            assert kwargs["resources"].device is None
            assert read_hf_export_request(request.checkpoint_path).status is HFExportStatus.IN_PROGRESS
            submissions.append(kwargs)
            return Job()
        yield SimpleNamespace(submit=submit)

    monkeypatch.setattr(export_hf_checkpoint, "open_iris_client", connect)
    monkeypatch.setattr(export_hf_checkpoint.IrisClient, "in_cluster", connect)
    return SimpleNamespace(
        spec=spec, events=events, uploads=uploads, failures=failures,
        store=store, submissions=submissions, save=save_export,
    )


@pytest.mark.parametrize("mode", [HFUploadMode.LATEST, HFUploadMode.ALL])
def test_gpu_conversion_finishes_before_cpu_publication(export_run, mode):
    run = export_run
    spec = replace(run.spec, request=replace(run.spec.request, hf_upload_mode=mode))
    export_hf_checkpoint.submit_requested_export(spec, ["convert"])
    assert run.events == ["gpu", "gpu-finished", "cpu"]
    assert run.uploads == ([""] if mode is HFUploadMode.LATEST else ["", "checkpoints/step_10"])
    assert read_hf_export_request(spec.request.checkpoint_path).status is HFExportStatus.COMPLETE
    assert len(run.submissions) == 1


def test_failed_hub_upload_retries_on_cpu_without_reconversion(export_run):
    run = export_run
    run.failures.append(True)
    with pytest.raises(subprocess.CalledProcessError):
        export_hf_checkpoint.submit_requested_export(run.spec, ["convert"])
    pending = read_hf_export_request(run.spec.request.checkpoint_path)
    assert pending.status is HFExportStatus.PENDING
    export_hf_checkpoint.submit_requested_export(replace(run.spec, request=pending), ["convert"])
    assert run.events == ["gpu", "gpu-finished", "cpu", "cpu"]
    assert run.uploads == [""]
    complete = read_hf_export_request(run.spec.request.checkpoint_path)
    assert complete.status is HFExportStatus.COMPLETE
    assert complete.attempts == 2


@pytest.mark.parametrize("inside_job", [False, True])
def test_cpu_upload_keeps_the_export_cluster_region(export_run, monkeypatch, inside_job):
    run = export_run
    monkeypatch.setattr(export_hf_checkpoint, "get_job_info", lambda: object() if inside_job else None)
    monkeypatch.setenv("IRIS_CONTROLLER_URL", "http://controller")
    export_hf_checkpoint.submit_requested_export(run.spec, ["convert"])
    constraints = run.submissions[0]["constraints"]
    assert region_constraint(["RNO2A"]) in constraints
    cluster_pin = Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value="cw-rno2a")
    assert (cluster_pin in constraints) is (not inside_job)
    assert run.uploads == [""]


def test_cpu_upload_pins_the_gcs_bucket_region(export_run, monkeypatch):
    run = export_run
    spec = replace(
        run.spec, cluster="marin", target_cluster=None,
        request=replace(run.spec.request, export_path="gs://fixture/exports"),
    )
    monkeypatch.setattr(export_hf_checkpoint, "get_bucket_location", lambda _path: "us-central2")
    export_hf_checkpoint.submit_requested_export(spec, ["convert"])
    assert run.submissions[0]["constraints"] == [region_constraint(["us-central2"])]
    assert run.uploads == [""]


def test_gpu_job_without_a_complete_export_never_submits_upload(export_run, monkeypatch):
    run = export_run
    monkeypatch.setattr(subprocess, "call", lambda *_args, **_kwargs: 0)
    with pytest.raises(RuntimeError, match="no safetensors weights"):
        export_hf_checkpoint.submit_requested_export(run.spec, ["convert"])
    assert run.submissions == []
    assert read_hf_export_request(run.spec.request.checkpoint_path).status is HFExportStatus.PENDING


def test_completed_object_store_export_needs_no_job(export_run):
    run = export_run
    run.save()
    spec = replace(run.spec, request=replace(run.spec.request, hf_hub_repo_id=None))
    export_hf_checkpoint.submit_requested_export(spec, ["convert"])
    assert run.events == []
    assert read_hf_export_request(spec.request.checkpoint_path).status is HFExportStatus.COMPLETE


def test_corrupt_export_is_reconverted_before_cpu_upload(export_run):
    run = export_run
    run.save()
    shard = policy_export_path(run.spec.request.export_path, run.spec.request.step) + "/model.safetensors"
    with run.store.open(shard, "ab") as destination:
        destination.write(b"truncated-or-replaced")
    export_hf_checkpoint.submit_requested_export(run.spec, ["convert"])
    assert run.events == ["gpu", "gpu-finished", "cpu"]
    assert run.uploads == [""]
