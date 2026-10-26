# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The identity OPD Iris jobs reach the controller with their time limits and no preemption retries.

The coordinator is submitted from the RL main's request, and the training job and the terminal export job from inside
it through SkyRL's Iris launcher, as in a run, to one in-process Iris controller. Each job's stored request is then
read back with GetJobStatus. The in-process controller has no federation peer, so the coordinator is submitted without
its cw-rno2a pin; a run's training and export jobs are submitted on cw-rno2a itself, unpinned, as here.
"""

import dataclasses
import getpass
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from fray.iris_backend import FrayIrisClient
from iris.client import context_state
from iris.client.client import IrisClient
from iris.cluster.client.job_info import set_job_info
from iris.cluster.config import (
    LocalSliceConfig,
    ScaleGroupConfig,
    ScaleGroupResources,
    SliceConfig,
    load_config,
    make_local_config,
)
from iris.cluster.local_cluster import LocalCluster
from iris.cluster.types import AcceleratorType, CapacityType
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.testing.config import DEFAULT_CONFIG
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.rl.cli import _coordinator_request

from experiments.post_training.identity_opd import launch
from experiments.post_training.identity_opd.test_identity_opd import _launch_doc, _step2_export


def _group(name: str, resources: ScaleGroupResources) -> ScaleGroupConfig:
    """A scale group that admits jobs of this shape and never starts a VM."""
    return ScaleGroupConfig(
        name=name,
        num_vms=1,
        buffer_slices=0,
        max_slices=0,
        resources=resources,
        slice_template=SliceConfig(local=LocalSliceConfig()),
    )


@pytest.fixture
def controller():
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups = {
        "cpu": _group(
            "cpu",
            ScaleGroupResources(
                cpu_millicores=8000,
                memory_bytes=64 * 1024**3,
                disk_bytes=256 * 1024**3,
                device_type=AcceleratorType.CPU,
                capacity_type=CapacityType.ON_DEMAND,
            ),
        ),
        "h100x8": _group(
            "h100x8",
            ScaleGroupResources(
                cpu_millicores=64000,
                memory_bytes=2048 * 1024**3,
                disk_bytes=4096 * 1024**3,
                device_type=AcceleratorType.GPU,
                device_variant="H100",
                device_count=8,
                capacity_type=CapacityType.ON_DEMAND,
            ),
        ),
    }
    cluster = LocalCluster(make_local_config(config))
    url = cluster.start()
    # The tokenless loopback caller is admin; let the submitting user use the production band, as the run's user can.
    client = ControllerServiceClientSync(address=url, timeout_ms=10000)
    client.set_user_budget(
        controller_pb2.Controller.SetUserBudgetRequest(
            user_id=getpass.getuser(), budget_limit=0, max_band=job_pb2.PRIORITY_BAND_PRODUCTION
        )
    )
    client.close()
    yield url
    cluster.close()


@pytest.fixture
def task_context():
    """Iris caches the task and context it runs in on first read; start and end each test outside any task."""
    set_job_info(None)
    token = context_state.set_context(None)
    yield
    context_state.reset_context(token)
    set_job_info(None)


def _submitted(controller: str, job_id: str) -> controller_pb2.Controller.LaunchJobRequest:
    client = ControllerServiceClientSync(address=controller, timeout_ms=10000)
    try:
        return client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=job_id)).request
    finally:
        client.close()


def _submit_coordinator(controller: str, arm: str, workspace: Path) -> str:
    unpinned = dataclasses.replace(launch.execution(arm), target_cluster=None, parent_cluster_config=None)
    step = ArtifactStep(
        name=f"checkpoints/{launch.EXPERIMENT_NAME}/{arm}",
        version="2026.10.26",
        artifact_type=Artifact,
        run=lambda config: None,
        build_config=lambda ctx: {},
        runtime_args={"skyrl_execution": unpinned},
    )
    _, request = _coordinator_request([step], launch.__name__, ("--arm", arm), 1, workspace, {})
    return str(
        FrayIrisClient.from_iris_client(IrisClient.remote(controller, workspace=workspace)).submit(request).job_id
    )


def _enter_task(controller: str, coordinator_id: str, workspace: Path, monkeypatch) -> Callable[[Path], str]:
    """Run inside the coordinator's task; return a submit that launches one document there and gives the new job id."""
    iris_backend = pytest.importorskip("cloud.iris.iris_backend")
    launch_config = pytest.importorskip("cloud.iris.launch_config")
    monkeypatch.setenv("IRIS_TASK_ID", f"{coordinator_id}/0")
    monkeypatch.setenv("IRIS_CONTROLLER_URL", controller)
    monkeypatch.setattr(iris_backend, "supervise_iris_job", lambda job: SimpleNamespace(exit_code=0))

    def submit(launch_doc: Path) -> str:
        before = {str(job.job_id) for job in IrisClient.remote(controller).list_jobs()}
        iris_backend.launch(
            iris_backend._iris_submission_state(launch_doc, launch_config.load_launch_config(launch_doc)), workspace
        )
        [job_id] = {str(job.job_id) for job in IrisClient.remote(controller).list_jobs()} - before
        return job_id

    return submit


@pytest.fixture
def workspace(tmp_path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text('[project]\nname = "workspace"\nversion = "0"\n')
    return workspace


@pytest.mark.parametrize("arm", ["dry", "train"])
def test_both_jobs_are_submitted_with_time_limits_and_no_preemption_retries(
    arm, controller, task_context, workspace, tmp_path, monkeypatch
):
    coordinator_id = _submit_coordinator(controller, arm, workspace)
    child_id = _enter_task(controller, coordinator_id, workspace, monkeypatch)(_launch_doc(arm, tmp_path))

    training_seconds = launch.JOB_TIMEOUT_SECONDS[arm]
    coordinator_seconds = training_seconds + launch.EXPORT_TIMEOUT_SECONDS + launch.COORDINATOR_MARGIN_SECONDS
    submitted_coordinator = _submitted(controller, coordinator_id)
    submitted_child = _submitted(controller, child_id)
    assert child_id.startswith(f"{coordinator_id}/")
    assert submitted_coordinator.timeout.milliseconds == coordinator_seconds * 1000
    assert submitted_child.timeout.milliseconds == training_seconds * 1000
    for submitted in (submitted_coordinator, submitted_child):
        assert submitted.max_retries_preemption == 0
        assert submitted.max_retries_failure == 0


@pytest.mark.parametrize("arm", ["dry", "train"])
def test_terminal_export_is_submitted_under_the_coordinator_with_its_limits(
    arm, controller, task_context, workspace, tmp_path, monkeypatch
):
    """The coordinator derives the terminal export with ``export_terminal_policy`` and submits it as its command does."""
    launch_config = pytest.importorskip("cloud.iris.launch_config")
    coordinator_id = _submit_coordinator(controller, arm, workspace)
    submit = _enter_task(controller, coordinator_id, workspace, monkeypatch)
    training_path = _launch_doc(arm, tmp_path)
    export_hf, _, specs = _step2_export(launch_config.load_launch_config(training_path), monkeypatch, tmp_path)

    export_hf.export_terminal_policy(training_path)
    [spec] = specs
    assert export_hf.build_command(spec)[-2:] == ["--config", spec.launch_config_path]
    export_id = submit(Path(spec.launch_config_path))

    submitted_export = _submitted(controller, export_id)
    assert export_id.startswith(f"{coordinator_id}/"), export_id
    assert "-export-2" in export_id.rsplit("/", 1)[-1]
    assert submitted_export.timeout.milliseconds == launch.EXPORT_TIMEOUT_SECONDS * 1000
    assert submitted_export.max_retries_preemption == 0
    assert submitted_export.max_retries_failure == 0
