# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cli.job — validation, placement policy, and bulk actions."""

import pytest
from click.testing import CliRunner
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cli.job import (
    build_job_constraints,
    build_job_summary,
    build_resources,
    kick,
    kill,
    run,
    stop,
    summary,
    wait,
)
from iris.client.client import IrisClient
from iris.client.workload_codec import job_status_from_proto, task_status_from_proto
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
    """--preemptible forces a preemptible=true constraint and bypasses the heuristic."""
    resources_proto = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB").to_proto()

    constraints = build_job_constraints(resources_proto, tpu_variants=[], replicas=1, preemptible=True)

    assert _preemptible_values(constraints) == ["true"]


def test_build_job_constraints_preemptible_false_emits_false_constraint():
    """--no-preemptible forces a preemptible=false constraint even for non-executor jobs."""
    resources_proto = build_resources(tpu=None, gpu=None, cpu=4.0, memory="16GB", disk="5GB").to_proto()

    constraints = build_job_constraints(resources_proto, tpu_variants=[], replicas=1, preemptible=False)

    assert _preemptible_values(constraints) == ["false"]


def test_build_job_constraints_preemptible_none_runs_heuristic():
    """Default (None) preserves the executor heuristic on small CPU jobs."""
    resources_proto = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB").to_proto()

    constraints = build_job_constraints(resources_proto, tpu_variants=[], replicas=1, preemptible=None)

    assert _preemptible_values(constraints) == ["false"]


def test_build_job_constraints_preemptible_true_overrides_heuristic():
    """Small CPU jobs normally auto-tag non-preemptible; --preemptible wins."""
    resources_proto = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB").to_proto()

    constraints = build_job_constraints(resources_proto, tpu_variants=[], replicas=1, preemptible=True)

    # Exactly one preemptible constraint, and it reflects the user's choice.
    assert _preemptible_values(constraints) == ["true"]


def test_build_job_constraints_target_cluster_appends_cluster_pin():
    """--target-cluster appends exactly one cluster EQ constraint naming the peer."""
    resources_proto = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB").to_proto()

    constraints = build_job_constraints(resources_proto, tpu_variants=[], replicas=1, target_cluster="peer-cluster")

    cluster_constraints = [c for c in constraints if c.key == CLUSTER_CONSTRAINT_KEY]
    assert len(cluster_constraints) == 1
    pin = cluster_constraints[0]
    assert pin.op == ConstraintOp.EQ
    assert pin.values[0].value == "peer-cluster"


def test_build_job_constraints_no_target_cluster_omits_cluster_pin():
    """Omitting --target-cluster appends no cluster constraint."""
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


def test_build_job_summary_includes_peak_memory_and_sorts_numerically():
    job = job_status_from_proto(
        _job_pb2.JobStatus(
            job_id="/u/j",
            name="train",
            state=_job_pb2.JOB_STATE_FAILED,
            exit_code=1,
            task_count=3,
            completed_count=3,
            task_state_counts={"succeeded": 2, "failed": 1},
        )
    )
    tasks = [
        task_status_from_proto(task)
        for task in [
            _task(10, _job_pb2.TASK_STATE_SUCCEEDED, peak_mb=2048, cur_mb=100, exit_code=0, duration_ms=65_000),
            _task(
                2, _job_pb2.TASK_STATE_FAILED, peak_mb=10_240, cur_mb=0, exit_code=137, duration_ms=5_000, error="OOM"
            ),
            _task(1, _job_pb2.TASK_STATE_SUCCEEDED, peak_mb=1024, cur_mb=50, exit_code=0, duration_ms=3_000),
        ]
    ]

    summary = build_job_summary(job, tasks)

    assert summary["job_id"] == "/u/j"
    assert summary["state"] == "failed"
    assert [t["index"] for t in summary["tasks"]] == ["1", "2", "10"]
    peaks = {t["index"]: t["memory_peak_mb"] for t in summary["tasks"]}
    assert peaks == {"1": 1024, "2": 10_240, "10": 2048}
    oom = next(t for t in summary["tasks"] if t["index"] == "2")
    assert oom["state"] == "failed"
    assert oom["exit_code"] == 137
    assert oom["error"] == "OOM"
    assert oom["duration_ms"] == 5_000


def test_build_job_summary_hides_exit_code_for_non_terminal_tasks():
    # The wire scalar default for exit_code is 0 — a RUNNING/BUILDING task must
    # not be reported as a clean exit=0 in the summary.
    job = job_status_from_proto(
        _job_pb2.JobStatus(job_id="/u/j", state=_job_pb2.JOB_STATE_RUNNING, task_count=3, completed_count=0)
    )
    running = task_status_from_proto(
        _task(0, _job_pb2.TASK_STATE_RUNNING, peak_mb=100, cur_mb=80, exit_code=0, duration_ms=1000)
    )
    building = task_status_from_proto(
        _job_pb2.TaskStatus(task_id="/u/j/1", state=_job_pb2.TASK_STATE_BUILDING, exit_code=0)
    )
    done = task_status_from_proto(
        _task(2, _job_pb2.TASK_STATE_SUCCEEDED, peak_mb=100, cur_mb=0, exit_code=0, duration_ms=1000)
    )
    summary = build_job_summary(job, [running, building, done])
    by_idx = {t["index"]: t for t in summary["tasks"]}
    assert by_idx["0"]["exit_code"] is None
    assert by_idx["1"]["exit_code"] is None
    assert by_idx["2"]["exit_code"] == 0


def test_job_summary_cli_shows_peak_memory(monkeypatch):
    job = job_status_from_proto(
        _job_pb2.JobStatus(job_id="/u/j", state=_job_pb2.JOB_STATE_FAILED, task_count=1, completed_count=1)
    )
    tasks = [
        task_status_from_proto(
            _task(
                0,
                _job_pb2.TASK_STATE_FAILED,
                peak_mb=9999,
                cur_mb=0,
                exit_code=137,
                duration_ms=1000,
                error="OOM",
            )
        )
    ]

    class FakeClient:
        def status(self, _job_id):
            return job

        def list_tasks(self, _job_id):
            return tasks

    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: FakeClient())
    result = CliRunner().invoke(summary, ["/u/j"], obj={"controller_url": "http://controller.test"})

    assert result.exit_code == 0, result.output
    assert "PEAK MEM" in result.output
    assert "10 GB" in result.output
    assert "137" in result.output
    assert "OOM" in result.output


def test_job_summary_cli_shows_active_backend_status(monkeypatch):
    job = job_status_from_proto(_job_pb2.JobStatus(job_id="/u/j", state=_job_pb2.JOB_STATE_RUNNING, task_count=1))
    task = task_status_from_proto(
        _job_pb2.TaskStatus(
            task_id="/u/j/0",
            state=_job_pb2.TASK_STATE_BUILDING,
            status_message='Kueue: excluded: resource "memory": 32',
        )
    )

    class FakeClient:
        def status(self, _job_id):
            return job

        def list_tasks(self, _job_id):
            return [task]

    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: FakeClient())
    result = CliRunner().invoke(summary, ["/u/j"], obj={"controller_url": "http://controller.test"})

    assert result.exit_code == 0, result.output
    assert 'Kueue: excluded: resource "memory": 32' in result.output


# Bulk-action target collection (query→act bridge for kick/stop/kill)
# ---------------------------------------------------------------------------


class _PrefixClusterClient:
    def __init__(self, active_job_id: str):
        self.active_job_id = active_job_id
        self.terminated: list[JobName] = []

    def list_jobs(self, *, query, **_kwargs):
        assert query.job_id_prefix == "/alice/"
        return [
            _job_pb2.JobStatus(job_id=self.active_job_id, state=_job_pb2.JOB_STATE_RUNNING),
            _job_pb2.JobStatus(job_id="/alice/done", state=_job_pb2.JOB_STATE_SUCCEEDED),
        ]

    def terminate_job(self, job_id):
        self.terminated.append(job_id)


def _kick_dry_run(args: list[str], input_text: str = ""):
    return CliRunner().invoke(
        kick,
        [*args, "--dry-run"],
        input=input_text,
        obj={"controller_url": "http://controller.test", "config": None, "credentials": None},
    )


def test_kick_stdin_drops_csv_header_and_extra_columns():
    # Exactly what `iris query -f csv "SELECT task_id, state FROM ..."` emits:
    # a header line with no leading slash, then id + trailing columns per row.
    result = _kick_dry_run(["--stdin"], "task_id,state\n/alice/job/0,3\n/bob/job/1,9\n")
    assert result.exit_code == 0, result.output
    assert "/alice/job/0" in result.output
    assert "/bob/job/1" in result.output


def test_kick_stdin_ignores_blank_and_non_id_lines():
    result = _kick_dry_run(["--stdin"], "/alice/job/0\n\n   \nNo jobs found.\n/bob/job\n")
    assert result.exit_code == 0, result.output
    assert "/alice/job/0" in result.output
    assert "/bob/job" in result.output
    assert "No jobs found." not in result.output


def test_kick_stdin_preserves_quoted_comma_and_space_ids():
    # JobName components may contain commas and spaces; iris query -f csv quotes
    # comma-bearing fields via csv.writer, so a real CSV parse must round-trip them.
    result = _kick_dry_run(["--stdin"], 'task_id,state\n"/alice/a,b/0",3\n/alice/my job/1,3\n')
    assert result.exit_code == 0, result.output
    assert "/alice/a,b/0" in result.output
    assert "/alice/my job/1" in result.output


def test_kick_stdin_skips_rows_with_empty_first_field():
    # A NULL first column (e.g. an unassigned current_worker_id) emits a leading
    # comma; the empty field must be skipped, not crash the whole action.
    result = _kick_dry_run(["--stdin"], ",3\n/alice/job/0,worker-1\n")
    assert result.exit_code == 0, result.output
    assert "would kick 1 target(s)" in result.output
    assert "/alice/job/0" in result.output


def test_kick_merges_positional_and_stdin_targets():
    result = _kick_dry_run(["/pos/job/0", "--stdin"], "/from/stdin/0\n")
    assert result.exit_code == 0, result.output
    assert "/pos/job/0" in result.output
    assert "/from/stdin/0" in result.output


def test_kick_dash_sentinel_reads_stdin():
    result = _kick_dry_run(["/pos/job/0", "-"], "/from/stdin/0\n")
    assert result.exit_code == 0, result.output
    assert "/pos/job/0" in result.output
    assert "/from/stdin/0" in result.output


def test_kick_without_stdin_uses_positional_targets():
    result = _kick_dry_run(["/a/b/0", "/a/c/0"])
    assert result.exit_code == 0, result.output
    assert "/a/b/0" in result.output
    assert "/a/c/0" in result.output


def test_kick_dry_run_lists_targets_without_sending():
    result = CliRunner().invoke(
        kick,
        ["--stdin", "--dry-run"],
        input="task_id\n/alice/job/0\n/bob/job/1\n",
        obj={"controller_url": "http://controller.test", "config": None, "credentials": None},
    )

    assert result.exit_code == 0, result.output
    assert "would kick 2 target(s) to preempted" in result.output
    assert "/alice/job/0" in result.output
    assert "/bob/job/1" in result.output


def test_kick_stdin_passes_collected_targets_to_client(monkeypatch):
    captured: dict[str, object] = {}

    class FakeClient:
        def kick_tasks(self, targets, *, desired_state, reason):
            captured["targets"] = targets
            captured["desired_state"] = desired_state
            captured["reason"] = reason
            return []

    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: FakeClient())

    result = CliRunner().invoke(
        kick,
        ["--stdin", "--state", "failed", "--reason", "drain"],
        input="/alice/job/0\n/bob/job/1\n",
        obj={"controller_url": "http://controller.test", "config": None, "credentials": None},
    )

    assert result.exit_code == 0, result.output
    assert captured["targets"] == ["/alice/job/0", "/bob/job/1"]
    assert captured["desired_state"] == _job_pb2.TASK_STATE_FAILED
    assert captured["reason"] == "drain"


def test_kick_no_targets_is_usage_error():
    result = CliRunner().invoke(kick, [], obj={"controller_url": "http://c.test", "config": None, "credentials": None})
    assert result.exit_code != 0
    assert "No targets given" in result.output


def test_stop_dry_run_lists_jobs_without_sending():
    result = CliRunner().invoke(
        stop,
        ["--stdin", "--dry-run"],
        input="job_id\n/alice/job\n",
        obj={"controller_url": "http://c.test", "config": None, "credentials": None},
    )
    assert result.exit_code == 0, result.output
    assert "would terminate 1 job(s)" in result.output
    assert "/alice/job" in result.output


def test_stop_prefix_dry_run_lists_matching_active_jobs_without_terminating(monkeypatch):
    cluster = _PrefixClusterClient("/alice/running")
    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: IrisClient(cluster))

    result = CliRunner().invoke(
        stop,
        ["--prefix", "/alice/", "--dry-run"],
        obj={"controller_url": "http://c.test", "config": None, "credentials": None},
    )

    assert result.exit_code == 0, result.output
    assert cluster.terminated == []
    assert "/alice/running" in result.output
    assert "/alice/done" not in result.output


@pytest.mark.parametrize("command", [stop, kill])
@pytest.mark.parametrize("match_args", [[], ["--exact"]], ids=["default", "explicit"])
def test_stop_commands_exact_match_terminates_only_named_job(monkeypatch, command, match_args):
    terminated: list[JobName] = []

    class FakeClient:
        def terminate(self, job_id):
            terminated.append(job_id)

        def terminate_prefix(self, prefix):
            matches = [prefix, JobName.from_wire(f"{prefix}-lp")]
            terminated.extend(matches)
            return matches

    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: FakeClient())

    result = CliRunner().invoke(
        command,
        [*match_args, "/alice/keep1"],
        obj={"controller_url": "http://c.test", "config": None, "credentials": None},
    )

    assert result.exit_code == 0, result.output
    assert terminated == [JobName.from_wire("/alice/keep1")]


def test_kill_prefix_terminates_matching_jobs(monkeypatch):
    terminated: list[JobName] = []

    class FakeClient:
        def terminate(self, job_id):
            terminated.append(job_id)

        def terminate_prefix(self, prefix):
            matches = [JobName.from_wire(prefix), JobName.from_wire(f"{prefix}-lp")]
            terminated.extend(matches)
            return matches

    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: FakeClient())

    result = CliRunner().invoke(
        kill,
        ["--prefix", "/alice/keep1"],
        obj={"controller_url": "http://c.test", "config": None, "credentials": None},
    )

    assert result.exit_code == 0, result.output
    assert terminated == [JobName.from_wire("/alice/keep1"), JobName.from_wire("/alice/keep1-lp")]


def test_stop_prefix_accepts_namespace_prefix(monkeypatch):
    cluster = _PrefixClusterClient("/alice/job")
    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: IrisClient(cluster))

    result = CliRunner().invoke(
        stop,
        ["--prefix", "/alice/"],
        obj={"controller_url": "http://c.test", "config": None, "credentials": None},
    )

    assert result.exit_code == 0, result.output
    assert cluster.terminated == [JobName.from_wire("/alice/job")]
    assert "/alice/job" in result.output


def test_kill_exact_miss_suggests_prefix_matches(monkeypatch):
    class FakeClient:
        def terminate(self, job_id):
            raise ConnectError(Code.NOT_FOUND, f"Job {job_id} not found")

        def list_jobs(self, *, prefix, limit):
            assert prefix == "/alice/keep1"
            assert limit == 5
            return [
                _job_pb2.JobStatus(job_id="/alice/keep1-lp"),
                _job_pb2.JobStatus(job_id="/alice/keep1-v2"),
            ]

    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: FakeClient())

    result = CliRunner().invoke(
        kill,
        ["/alice/keep1"],
        obj={"controller_url": "http://c.test", "config": None, "credentials": None},
    )

    assert result.exit_code != 0
    assert "No job named '/alice/keep1'" in result.output
    assert "Did you mean: /alice/keep1-lp, /alice/keep1-v2?" in result.output
