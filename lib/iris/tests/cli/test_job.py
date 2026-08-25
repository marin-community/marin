# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Iris Job submission and operator commands."""

import pytest
from click.testing import CliRunner
from iris.cli.job import (
    build_job_constraints,
    build_resources,
    cancel,
    complete,
    describe,
    list_jobs,
    run,
    wait,
)
from iris.client.client import IrisClient
from iris.client.workload_codec import job_status_from_proto
from iris.cluster.config import IrisClusterConfig, ScaleGroupConfig, WorkerSettings
from iris.cluster.constraints import (
    CLUSTER_CONSTRAINT_KEY,
    Constraint,
    ConstraintOp,
    WellKnownAttribute,
    infer_preemptible_constraint,
    preemptible_constraint,
    region_constraint,
)
from iris.cluster.types import JobName
from iris.rpc import job_pb2 as _job_pb2


def _make_config_with_zones(zones: list[str]) -> IrisClusterConfig:
    """Build a minimal IrisClusterConfig with scale groups for the given zones."""
    scale_groups: dict[str, ScaleGroupConfig] = {}
    for zone in zones:
        region = zone.rsplit("-", 1)[0]
        scale_groups[f"sg-{zone}"] = ScaleGroupConfig(worker=WorkerSettings(attributes={"zone": zone, "region": region}))
    return IrisClusterConfig(scale_groups=scale_groups)


def _run_cli(args: list[str], *, config: IrisClusterConfig | None = None):
    return CliRunner().invoke(
        run,
        [*args, "--no-wait", "--", "echo", "ok"],
        obj={"controller_url": "http://controller.test", "config": config, "credentials": None},
    )


def test_validate_region_zone_valid_region(recorded_job_submissions):
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    result = _run_cli(["--region", "us-central2"], config=config)
    assert result.exit_code == 0, result.output
    assert len(recorded_job_submissions) == 1


def test_validate_region_zone_valid_zone(recorded_job_submissions):
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    result = _run_cli(["--zone", "europe-west4-a"], config=config)
    assert result.exit_code == 0, result.output
    assert len(recorded_job_submissions) == 1


def test_validate_region_zone_invalid_region_raises():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    result = _run_cli(["--region", "eu-west4"], config=config)
    assert result.exit_code != 0
    assert "eu-west4" in result.output
    assert "not a known region" in result.output


def test_validate_region_zone_invalid_region_suggests_closest():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    result = _run_cli(["--region", "eu-west4"], config=config)
    assert result.exit_code != 0
    assert "Did you mean 'europe-west4'" in result.output


def test_validate_region_zone_invalid_zone_raises():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    result = _run_cli(["--zone", "us-central2-a"], config=config)
    assert result.exit_code != 0
    assert "us-central2-a" in result.output
    assert "not a known zone" in result.output


def test_validate_region_zone_invalid_zone_suggests_closest():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    result = _run_cli(["--zone", "us-central2-a"], config=config)
    assert result.exit_code != 0
    assert "Did you mean 'us-central2-b'" in result.output


def test_job_run_accepts_unresolved_placement_without_cluster_metadata(recorded_job_submissions):
    # Without cluster metadata, accepting an unresolved placement constraint is the public contract.
    result = _run_cli(["--region", "nonexistent", "--zone", "nonexistent"])
    assert result.exit_code == 0, result.output
    assert len(recorded_job_submissions) == 1


def test_job_run_accepts_unconstrained_placement_with_cluster_metadata(recorded_job_submissions):
    config = _make_config_with_zones(["us-central2-b"])
    # A configured cluster does not require callers to constrain placement.
    result = _run_cli([], config=config)
    assert result.exit_code == 0, result.output
    assert len(recorded_job_submissions) == 1


@pytest.fixture
def recorded_bundle_exclude(monkeypatch):
    """Capture the ``bundle_exclude`` passed to ``IrisClient.remote`` by ``iris job run``."""
    captured: dict[str, object] = {}

    class FakeJob:
        job_id = JobName.from_wire("/test-user/test-job")

    class FakeClient:
        def submit(self, **kwargs):
            return FakeJob()

    def fake_remote(*args, **kwargs):
        captured["bundle_exclude"] = kwargs.get("bundle_exclude")
        return FakeClient()

    monkeypatch.setattr("iris.cli.job.IrisClient.remote", fake_remote)
    return captured


def test_exclude_options_become_one_or_ed_bundle_regex(recorded_bundle_exclude):
    # Each --exclude flag contributes an independent alternative; a path matching any
    # one is dropped, and an unrelated path is kept.
    result = _run_cli(["--exclude", r"^docs/", "--exclude", r"^data/"])
    assert result.exit_code == 0, result.output
    pattern = recorded_bundle_exclude["bundle_exclude"]
    assert pattern.search("docs/guide.md")
    assert pattern.search("data/big.csv")
    assert not pattern.search("src/main.py")


def test_no_exclude_leaves_bundle_exclude_unset(recorded_bundle_exclude):
    result = _run_cli([])
    assert result.exit_code == 0, result.output
    assert recorded_bundle_exclude["bundle_exclude"] is None


# ---------------------------------------------------------------------------
# Executor heuristic tests (mirrors the logic in run_iris_job)
# ---------------------------------------------------------------------------


def test_executor_heuristic_small_cpu_job_gets_non_preemptible():
    resources = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB")
    resources_proto = resources.to_proto()
    replicas = 1
    constraints: list[Constraint] = []

    preemptible = infer_preemptible_constraint(resources_proto, replicas, constraints)
    assert preemptible is not None
    assert preemptible.key == WellKnownAttribute.PREEMPTIBLE
    assert preemptible.values[0].value == "false"


def test_executor_heuristic_skipped_for_gpu_job():
    resources = build_resources(tpu=None, gpu="H100", cpu=0.5, memory="1GB", disk="5GB")
    resources_proto = resources.to_proto()
    replicas = 1
    constraints: list[Constraint] = []

    preemptible = infer_preemptible_constraint(resources_proto, replicas, constraints)
    assert preemptible is None


def test_executor_heuristic_skipped_for_large_cpu_job():
    resources = build_resources(tpu=None, gpu=None, cpu=4.0, memory="16GB", disk="5GB")
    resources_proto = resources.to_proto()
    replicas = 1
    constraints: list[Constraint] = []

    preemptible = infer_preemptible_constraint(resources_proto, replicas, constraints)
    assert preemptible is None


def test_executor_heuristic_skipped_when_user_sets_preemptible():
    resources = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB")
    resources_proto = resources.to_proto()
    replicas = 1
    constraints: list[Constraint] = [preemptible_constraint(True)]

    preemptible = infer_preemptible_constraint(resources_proto, replicas, constraints)
    assert preemptible is None


def test_executor_heuristic_with_region_constraint():
    resources = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB")
    resources_proto = resources.to_proto()
    replicas = 1
    constraints: list[Constraint] = [region_constraint(["us-central2"])]

    preemptible = infer_preemptible_constraint(resources_proto, replicas, constraints)
    assert preemptible is not None
    assert preemptible.values[0].value == "false"


# ---------------------------------------------------------------------------
# build_job_constraints — --preemptible / --no-preemptible wiring (#4540)
# ---------------------------------------------------------------------------


def _preemptible_values(constraints: list[Constraint]) -> list[str]:
    return [c.values[0].value for c in constraints if c.key == WellKnownAttribute.PREEMPTIBLE]


def test_build_job_constraints_preemptible_true_emits_true_constraint():
    resources_proto = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB").to_proto()

    constraints = build_job_constraints(resources_proto, tpu_variants=[], replicas=1, preemptible=True)

    assert _preemptible_values(constraints) == ["true"]


def test_build_job_constraints_preemptible_false_emits_false_constraint():
    resources_proto = build_resources(tpu=None, gpu=None, cpu=4.0, memory="16GB", disk="5GB").to_proto()

    constraints = build_job_constraints(resources_proto, tpu_variants=[], replicas=1, preemptible=False)

    assert _preemptible_values(constraints) == ["false"]


def test_build_job_constraints_preemptible_none_runs_heuristic():
    resources_proto = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB").to_proto()

    constraints = build_job_constraints(resources_proto, tpu_variants=[], replicas=1, preemptible=None)

    assert _preemptible_values(constraints) == ["false"]


def test_build_job_constraints_preemptible_true_overrides_heuristic():
    resources_proto = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB").to_proto()

    constraints = build_job_constraints(resources_proto, tpu_variants=[], replicas=1, preemptible=True)

    assert _preemptible_values(constraints) == ["true"]


def test_build_job_constraints_target_cluster_appends_cluster_pin():
    resources_proto = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB").to_proto()

    constraints = build_job_constraints(resources_proto, tpu_variants=[], replicas=1, target_cluster="peer-cluster")

    cluster_constraints = [c for c in constraints if c.key == CLUSTER_CONSTRAINT_KEY]
    assert len(cluster_constraints) == 1
    pin = cluster_constraints[0]
    assert pin.op == ConstraintOp.EQ
    assert pin.values[0].value == "peer-cluster"


def test_build_job_constraints_no_target_cluster_omits_cluster_pin():
    resources_proto = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB").to_proto()

    constraints = build_job_constraints(resources_proto, tpu_variants=[], replicas=1, target_cluster=None)

    assert [c for c in constraints if c.key == CLUSTER_CONSTRAINT_KEY] == []


def test_job_run_cli_accepts_task_image_override(monkeypatch):
    captured: dict[str, object] = {}

    class FakeJob:
        job_id = "test-job"

    class FakeClient:
        def submit(self, **kwargs):
            captured.update(kwargs)
            return FakeJob()

    def fake_remote(controller_url, *, workspace, credentials=None, bundle_exclude=None):
        captured["controller_url"] = controller_url
        captured["workspace"] = workspace
        captured["credentials"] = credentials
        return FakeClient()

    monkeypatch.setattr("iris.cli.job.IrisClient.remote", fake_remote)

    result = CliRunner().invoke(
        run,
        [
            "--task-image",
            "ghcr.io/marin-community/iris-task-cuda-devel:test",
            "--no-wait",
            "--",
            "python",
            "train.py",
        ],
        obj={"controller_url": "http://controller.test", "config": None, "credentials": None},
    )

    assert result.exit_code == 0, result.output
    assert captured["task_image"] == "ghcr.io/marin-community/iris-task-cuda-devel:test"
    assert captured["controller_url"] == "http://controller.test"
    assert captured["entrypoint"].command == ["python", "train.py"]


@pytest.mark.parametrize(
    ("state", "expected_state", "expected_exit_code"),
    [
        (_job_pb2.JOB_STATE_SUCCEEDED, "succeeded", 0),
        (_job_pb2.JOB_STATE_FAILED, "failed", 1),
    ],
)
def test_job_wait_reports_terminal_state_and_exit_status(
    monkeypatch,
    state: _job_pb2.JobState,
    expected_state: str,
    expected_exit_code: int,
) -> None:
    class WaitClusterClient:
        def wait_for_job(self, job_id, _timeout, _poll_interval):
            return _job_pb2.JobStatus(job_id=job_id.to_wire(), state=state)

    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: IrisClient(WaitClusterClient()))

    result = CliRunner().invoke(
        wait,
        ["/alice/training-run"],
        obj={"controller_url": "http://controller.test", "config": None, "credentials": None},
    )

    assert result.exit_code == expected_exit_code
    assert result.output == f"{expected_state}\n"


# --tpu multi-variant parsing
# ---------------------------------------------------------------------------


def test_tpu_multi_variant_parsing(recorded_job_submissions):
    result = _run_cli(
        ["--enable-extra-resources", "--tpu", " v6e-4 , v5litepod-4 , v5p-8 "],
    )
    assert result.exit_code == 0, result.output
    submission = recorded_job_submissions[0]
    assert submission["resources"].device.tpu.variant == "v6e-4"
    device_constraint = next(c for c in submission["constraints"] if c.key == WellKnownAttribute.DEVICE_VARIANT)
    assert [value.value for value in device_constraint.values] == ["v6e-4", "v5litepod-4", "v5p-8"]

    empty = _run_cli(["--enable-extra-resources", "--tpu", ", ,"])
    assert empty.exit_code != 0
    assert "at least one" in empty.output

    mismatched = _run_cli(["--enable-extra-resources", "--tpu", "v5p-8,v5p-16"])
    assert mismatched.exit_code != 0
    assert "vm_count" in mismatched.output


# ---------------------------------------------------------------------------
# validate_extra_resources tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["--tpu", "v5litepod-16"], "--tpu requires --enable-extra-resources"),
        (["--gpu", "H100x8"], "--gpu requires --enable-extra-resources"),
        (["--memory", "4GB"], "--memory 4GB"),
        (["--disk", "10GB"], "--disk 10GB"),
    ],
)
def test_validate_extra_resources(args, error):
    result = _run_cli(args)
    assert result.exit_code != 0
    assert error in result.output


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--memory", "3900MB"],
        ["--disk", "9900MB"],
        ["--enable-extra-resources", "--tpu", "v5litepod-16"],
        ["--enable-extra-resources", "--memory", "64GB"],
        ["--enable-extra-resources", "--disk", "100GB"],
    ],
)
def test_validate_extra_resources_accepts_supported_requests(args, recorded_job_submissions):
    result = _run_cli(args)
    assert result.exit_code == 0, result.output
    assert len(recorded_job_submissions) == 1


def _task(index: int, state, *, peak_mb: int, cur_mb: int, exit_code: int, duration_ms: int, error: str = ""):
    t = _job_pb2.TaskStatus(
        task_id=f"/u/j/{index}",
        state=state,
        exit_code=exit_code,
        error=error,
    )
    t.resource_usage.memory_peak_mb = peak_mb
    t.resource_usage.memory_mb = cur_mb
    t.started_at.epoch_ms = 1_000_000
    t.finished_at.epoch_ms = 1_000_000 + duration_ms
    return t


class _SummaryTransport:
    def __init__(self, job: _job_pb2.JobStatus, tasks: tuple[_job_pb2.TaskStatus, ...]):
        self.job = job
        self.tasks = tasks

    def get_job_status(self, _job_id):
        return self.job

    def list_tasks(self, _job_id):
        return list(self.tasks)


def _summary_statuses(job: _job_pb2.JobStatus, *tasks: _job_pb2.TaskStatus):
    client = IrisClient(_SummaryTransport(job, tasks))
    job_id = JobName.from_wire(job.job_id)
    return client.job_status(job_id), client.list_tasks(job_id)


def _description_task_rows(output: str) -> list[list[str]]:
    rows = [line.split() for line in output.splitlines()]
    return [row for row in rows if row and row[0].isdigit()]


def test_job_describe_cli_includes_peak_memory_and_sorts_numerically(monkeypatch):
    job, tasks = _summary_statuses(
        _job_pb2.JobStatus(
            job_id="/u/j",
            name="train",
            state=_job_pb2.JOB_STATE_FAILED,
            exit_code=1,
            task_count=3,
            completed_count=3,
            task_state_counts={"succeeded": 2, "failed": 1},
        ),
        _task(10, _job_pb2.TASK_STATE_SUCCEEDED, peak_mb=2048, cur_mb=100, exit_code=0, duration_ms=65_000),
        _task(2, _job_pb2.TASK_STATE_FAILED, peak_mb=10_240, cur_mb=0, exit_code=137, duration_ms=5_000, error="OOM"),
        _task(1, _job_pb2.TASK_STATE_SUCCEEDED, peak_mb=1024, cur_mb=50, exit_code=0, duration_ms=3_000),
    )

    class FakeClient:
        def job_status(self, _job_id):
            return job

        def list_tasks(self, _job_id):
            return tasks

    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: FakeClient())
    result = CliRunner().invoke(describe, ["/u/j"], obj={"controller_url": "http://controller.test"})

    assert result.exit_code == 0, result.output
    task_lines = _description_task_rows(result.output)
    assert [line[0] for line in task_lines] == ["1", "2", "10"]
    assert task_lines[1][1:3] == ["failed", "137"]
    assert "10.24 GB" in result.output
    assert "OOM" in result.output


def test_job_describe_cli_hides_exit_code_for_non_terminal_tasks(monkeypatch):
    # The wire scalar default for exit_code is 0 — a RUNNING/BUILDING task must
    # not be reported as a clean exit=0 in the description.
    job, tasks = _summary_statuses(
        _job_pb2.JobStatus(job_id="/u/j", state=_job_pb2.JOB_STATE_RUNNING, task_count=3, completed_count=0),
        _task(0, _job_pb2.TASK_STATE_RUNNING, peak_mb=100, cur_mb=80, exit_code=0, duration_ms=1000),
        _job_pb2.TaskStatus(task_id="/u/j/1", state=_job_pb2.TASK_STATE_BUILDING, exit_code=0),
        _task(2, _job_pb2.TASK_STATE_SUCCEEDED, peak_mb=100, cur_mb=0, exit_code=0, duration_ms=1000),
    )

    class FakeClient:
        def job_status(self, _job_id):
            return job

        def list_tasks(self, _job_id):
            return tasks

    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: FakeClient())
    result = CliRunner().invoke(describe, ["/u/j"], obj={"controller_url": "http://controller.test"})

    assert result.exit_code == 0, result.output
    task_lines = {row[0]: row for row in _description_task_rows(result.output)}
    assert task_lines["0"][2] == "-"
    assert task_lines["1"][2] == "-"
    assert task_lines["2"][2] == "0"


def test_job_describe_cli_shows_active_backend_status(monkeypatch):
    job, tasks = _summary_statuses(
        _job_pb2.JobStatus(job_id="/u/j", state=_job_pb2.JOB_STATE_RUNNING, task_count=1),
        _job_pb2.TaskStatus(
            task_id="/u/j/0",
            state=_job_pb2.TASK_STATE_BUILDING,
            status_message='Kueue: excluded: resource "memory": 32',
        ),
    )

    class FakeClient:
        def job_status(self, _job_id):
            return job

        def list_tasks(self, _job_id):
            return tasks

    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: FakeClient())
    result = CliRunner().invoke(describe, ["/u/j"], obj={"controller_url": "http://controller.test"})

    assert result.exit_code == 0, result.output
    assert 'Kueue: excluded: resource "memory": 32' in result.output


def test_job_list_cli_reports_why_a_building_job_is_waiting(monkeypatch):
    """A BUILDING job has no pending_reason, so its backend status fills the REASON column."""
    job = job_status_from_proto(
        _job_pb2.JobStatus(
            job_id="/u/gang",
            state=_job_pb2.JOB_STATE_BUILDING,
            status_message="SchedulingGated: waiting for Kueue quota",
        )
    )

    class FakeClient:
        def list_jobs(self, **_kwargs):
            return [job]

    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: FakeClient())
    result = CliRunner().invoke(list_jobs, [], obj={"controller_url": "http://controller.test"})

    assert result.exit_code == 0, result.output
    assert "REASON" in result.output
    assert "SchedulingGated: waiting for Kueue quota" in result.output


class _JobActionClusterClient:
    def __init__(self):
        self.cancelled: list[JobName] = []
        self.completed: list[JobName] = []

    def list_jobs(self, *, query, **_kwargs):
        jobs = [
            _job_pb2.JobStatus(job_id="/alice/running", state=_job_pb2.JOB_STATE_RUNNING),
            _job_pb2.JobStatus(job_id="/alice/done", state=_job_pb2.JOB_STATE_SUCCEEDED),
            _job_pb2.JobStatus(job_id="/bob/running", state=_job_pb2.JOB_STATE_RUNNING),
        ]
        return [job for job in jobs if job.job_id.startswith(query.job_id_prefix)]

    def terminate_job(self, job_id):
        self.cancelled.append(job_id)

    def complete_job(self, job_id):
        self.completed.append(job_id)


def test_job_cancel_combines_positional_and_csv_stdin_targets():
    result = CliRunner().invoke(
        cancel,
        ["/alice/first", "--stdin", "--dry-run"],
        input='job_id,state\n"/alice/second,run",3\n',
        obj={"controller_url": "http://controller.test", "config": None, "credentials": None},
    )

    assert result.exit_code == 0, result.output
    assert "/alice/first" in result.output
    assert "/alice/second,run" in result.output


def test_job_cancel_prefix_cancels_only_active_jobs(monkeypatch):
    cluster = _JobActionClusterClient()
    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: IrisClient(cluster))

    result = CliRunner().invoke(
        cancel,
        ["--prefix", "/alice/"],
        obj={"controller_url": "http://controller.test", "config": None, "credentials": None},
    )

    assert result.exit_code == 0, result.output
    assert cluster.cancelled == [JobName.from_wire("/alice/running")]
    assert "/alice/running" in result.output
    assert "/alice/done" not in result.output


def test_job_complete_targets_selected_job(monkeypatch):
    cluster = _JobActionClusterClient()
    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: IrisClient(cluster))

    result = CliRunner().invoke(
        complete,
        ["/alice/running"],
        obj={"controller_url": "http://controller.test", "config": None, "credentials": None},
    )

    assert result.exit_code == 0, result.output
    assert cluster.completed == [JobName.from_wire("/alice/running")]
