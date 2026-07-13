# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cli.job — validate_region_zone, executor heuristic, and related CLI validation."""

import io

import click
import pytest
from click.testing import CliRunner
from iris.cli.job import (
    _collect_targets,
    _parse_tpu_alternatives,
    _read_targets_from_stdin,
    _render_job_summary_text,
    build_job_constraints,
    build_job_summary,
    build_resources,
    build_tpu_alternatives,
    kick,
    run,
    stop,
    validate_extra_resources,
    validate_region_zone,
)
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
from iris.rpc import job_pb2 as _job_pb2


def _make_config_with_zones(zones: list[str]) -> IrisClusterConfig:
    """Build a minimal IrisClusterConfig with scale groups for the given zones."""
    scale_groups: dict[str, ScaleGroupConfig] = {}
    for zone in zones:
        region = zone.rsplit("-", 1)[0]
        scale_groups[f"sg-{zone}"] = ScaleGroupConfig(worker=WorkerSettings(attributes={"zone": zone, "region": region}))
    return IrisClusterConfig(scale_groups=scale_groups)


def test_validate_region_zone_valid_region():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    validate_region_zone(("us-central2",), None, config)


def test_validate_region_zone_valid_zone():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    validate_region_zone(None, "europe-west4-a", config)


def test_validate_region_zone_invalid_region_raises():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    with pytest.raises(click.BadParameter, match=r"eu-west4.*not a known region"):
        validate_region_zone(("eu-west4",), None, config)


def test_validate_region_zone_invalid_region_suggests_closest():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    with pytest.raises(click.BadParameter, match="Did you mean 'europe-west4'"):
        validate_region_zone(("eu-west4",), None, config)


def test_validate_region_zone_invalid_zone_raises():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    with pytest.raises(click.BadParameter, match=r"us-central2-a.*not a known zone"):
        validate_region_zone(None, "us-central2-a", config)


def test_validate_region_zone_invalid_zone_suggests_closest():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    with pytest.raises(click.BadParameter, match="Did you mean 'us-central2-b'"):
        validate_region_zone(None, "us-central2-a", config)


def test_validate_region_zone_no_config_skips():
    validate_region_zone(("nonexistent",), "nonexistent", None)


def test_validate_region_zone_no_constraints_skips():
    config = _make_config_with_zones(["us-central2-b"])
    validate_region_zone(None, None, config)


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

    def fake_remote(controller_url, *, workspace, credentials=None):
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


# --tpu multi-variant parsing
# ---------------------------------------------------------------------------


def test_tpu_multi_variant_parsing():
    # Single variant
    primary, alts = _parse_tpu_alternatives("v6e-4")
    assert (primary, alts) == ("v6e-4", [])

    # Comma-separated list: first is primary, rest are alternatives; whitespace stripped
    primary, alts = _parse_tpu_alternatives(" v6e-4 , v5litepod-4 , v5p-8 ")
    assert (primary, alts) == ("v6e-4", ["v5litepod-4", "v5p-8"])

    # Empty / garbage input rejected
    with pytest.raises(click.BadParameter, match="at least one"):
        _parse_tpu_alternatives(", ,")

    # Mismatched vm_count across variants rejected
    with pytest.raises(click.BadParameter, match="vm_count"):
        _parse_tpu_alternatives("v5p-8,v5p-16")

    # build_tpu_alternatives: None → [], multi-variant → flat list
    assert build_tpu_alternatives(None) == []
    assert build_tpu_alternatives("v6e-4,v5litepod-4,v5p-8") == ["v6e-4", "v5litepod-4", "v5p-8"]

    # build_resources picks the first variant as the canonical TPU type
    spec = build_resources(tpu="v6e-4,v5litepod-4,v5p-8", gpu=None, cpu=8.0, memory="32GB", disk="50GB")
    assert spec.device.tpu.variant == "v6e-4"


# ---------------------------------------------------------------------------
# validate_extra_resources tests
# ---------------------------------------------------------------------------


def test_validate_extra_resources():
    # Normal CPU-only job passes without the flag.
    validate_extra_resources(tpu=None, gpu=None, memory="1GB", disk="5GB", enable_extra_resources=False)

    # TPU and GPU blocked without the flag; error names the coordinator pattern.
    with pytest.raises(click.UsageError, match="--tpu requires --enable-extra-resources"):
        validate_extra_resources(tpu="v5litepod-16", gpu=None, memory="1GB", disk="5GB", enable_extra_resources=False)
    with pytest.raises(click.UsageError, match="--gpu requires --enable-extra-resources"):
        validate_extra_resources(tpu=None, gpu="H100x8", memory="1GB", disk="5GB", enable_extra_resources=False)
    with pytest.raises(click.UsageError, match="coordinator"):
        validate_extra_resources(tpu="v5litepod-16", gpu=None, memory="1GB", disk="5GB", enable_extra_resources=False)

    # Memory threshold: >= 4 GB blocked, < 4 GB allowed.
    with pytest.raises(click.UsageError, match=r"--memory 4GB.*--enable-extra-resources"):
        validate_extra_resources(tpu=None, gpu=None, memory="4GB", disk="5GB", enable_extra_resources=False)
    validate_extra_resources(tpu=None, gpu=None, memory="3900MB", disk="5GB", enable_extra_resources=False)

    # Disk threshold: >= 10 GB blocked, < 10 GB allowed.
    with pytest.raises(click.UsageError, match=r"--disk 10GB.*--enable-extra-resources"):
        validate_extra_resources(tpu=None, gpu=None, memory="1GB", disk="10GB", enable_extra_resources=False)
    validate_extra_resources(tpu=None, gpu=None, memory="1GB", disk="9900MB", enable_extra_resources=False)

    # --enable-extra-resources bypasses all checks.
    validate_extra_resources(tpu="v5litepod-16", gpu=None, memory="1GB", disk="5GB", enable_extra_resources=True)
    validate_extra_resources(tpu=None, gpu=None, memory="64GB", disk="5GB", enable_extra_resources=True)
    validate_extra_resources(tpu=None, gpu=None, memory="1GB", disk="100GB", enable_extra_resources=True)


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
    job = _job_pb2.JobStatus(
        job_id="/u/j",
        name="train",
        state=_job_pb2.JOB_STATE_FAILED,
        exit_code=1,
        task_count=3,
        completed_count=3,
        task_state_counts={"succeeded": 2, "failed": 1},
    )
    tasks = [
        _task(10, _job_pb2.TASK_STATE_SUCCEEDED, peak_mb=2048, cur_mb=100, exit_code=0, duration_ms=65_000),
        _task(2, _job_pb2.TASK_STATE_FAILED, peak_mb=10_240, cur_mb=0, exit_code=137, duration_ms=5_000, error="OOM"),
        _task(1, _job_pb2.TASK_STATE_SUCCEEDED, peak_mb=1024, cur_mb=50, exit_code=0, duration_ms=3_000),
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
    # Proto scalar default for exit_code is 0 — a RUNNING/BUILDING task must
    # not be reported as a clean exit=0 in the summary.
    job = _job_pb2.JobStatus(job_id="/u/j", state=_job_pb2.JOB_STATE_RUNNING, task_count=3, completed_count=0)
    running = _task(0, _job_pb2.TASK_STATE_RUNNING, peak_mb=100, cur_mb=80, exit_code=0, duration_ms=1000)
    building = _job_pb2.TaskStatus(task_id="/u/j/1", state=_job_pb2.TASK_STATE_BUILDING, exit_code=0)
    done = _task(2, _job_pb2.TASK_STATE_SUCCEEDED, peak_mb=100, cur_mb=0, exit_code=0, duration_ms=1000)
    summary = build_job_summary(job, [running, building, done])
    by_idx = {t["index"]: t for t in summary["tasks"]}
    assert by_idx["0"]["exit_code"] is None
    assert by_idx["1"]["exit_code"] is None
    assert by_idx["2"]["exit_code"] == 0


def test_render_job_summary_text_shows_peak_memory():
    job = _job_pb2.JobStatus(job_id="/u/j", state=_job_pb2.JOB_STATE_FAILED, task_count=1, completed_count=1)
    tasks = [_task(0, _job_pb2.TASK_STATE_FAILED, peak_mb=9999, cur_mb=0, exit_code=137, duration_ms=1000, error="OOM")]
    text = _render_job_summary_text(build_job_summary(job, tasks))
    assert "PEAK MEM" in text
    # 9999 MB is formatted as "10 GB" by humanfriendly
    assert "10 GB" in text
    assert "137" in text
    assert "OOM" in text


# Bulk-action target collection (query→act bridge for kick/stop/kill)
# ---------------------------------------------------------------------------


def test_read_targets_from_stdin_drops_csv_header_and_extra_columns(monkeypatch):
    # Exactly what `iris query -f csv "SELECT task_id, state FROM ..."` emits:
    # a header line with no leading slash, then id + trailing columns per row.
    stdin = io.StringIO("task_id,state\n/alice/job/0,3\n/bob/job/1,9\n")
    monkeypatch.setattr("iris.cli.job.sys.stdin", stdin)
    assert _read_targets_from_stdin() == ["/alice/job/0", "/bob/job/1"]


def test_read_targets_from_stdin_ignores_blank_and_non_id_lines(monkeypatch):
    stdin = io.StringIO("/alice/job/0\n\n   \nNo jobs found.\n/bob/job\n")
    monkeypatch.setattr("iris.cli.job.sys.stdin", stdin)
    assert _read_targets_from_stdin() == ["/alice/job/0", "/bob/job"]


def test_read_targets_from_stdin_preserves_quoted_comma_and_space_ids(monkeypatch):
    # JobName components may contain commas and spaces; iris query -f csv quotes
    # comma-bearing fields via csv.writer, so a real CSV parse must round-trip them.
    stdin = io.StringIO('task_id,state\n"/alice/a,b/0",3\n/alice/my job/1,3\n')
    monkeypatch.setattr("iris.cli.job.sys.stdin", stdin)
    assert _read_targets_from_stdin() == ["/alice/a,b/0", "/alice/my job/1"]


def test_read_targets_from_stdin_skips_rows_with_empty_first_field(monkeypatch):
    # A NULL first column (e.g. an unassigned current_worker_id) emits a leading
    # comma; the empty field must be skipped, not crash the whole action.
    stdin = io.StringIO(",3\n/alice/job/0,worker-1\n")
    monkeypatch.setattr("iris.cli.job.sys.stdin", stdin)
    assert _read_targets_from_stdin() == ["/alice/job/0"]


def test_collect_targets_merges_positional_and_stdin(monkeypatch):
    monkeypatch.setattr("iris.cli.job.sys.stdin", io.StringIO("/from/stdin/0\n"))
    assert _collect_targets(("/pos/job/0",), use_stdin=True) == ["/pos/job/0", "/from/stdin/0"]


def test_collect_targets_dash_sentinel_reads_stdin(monkeypatch):
    monkeypatch.setattr("iris.cli.job.sys.stdin", io.StringIO("/from/stdin/0\n"))
    # '-' is consumed as the stdin sentinel, not passed through as a target.
    assert _collect_targets(("/pos/job/0", "-"), use_stdin=False) == ["/pos/job/0", "/from/stdin/0"]


def test_collect_targets_no_stdin_returns_positional_only():
    assert _collect_targets(("/a/b/0", "/a/c/0"), use_stdin=False) == ["/a/b/0", "/a/c/0"]


def test_kick_dry_run_lists_targets_without_sending(monkeypatch):
    def _boom(_ctx):
        raise AssertionError("dry-run must not open a client or send an RPC")

    monkeypatch.setattr("iris.cli.job._remote_client", _boom)

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


def test_kick_no_targets_is_usage_error(monkeypatch):
    monkeypatch.setattr("iris.cli.job._remote_client", lambda _ctx: pytest.fail("should not reach client"))
    result = CliRunner().invoke(kick, [], obj={"controller_url": "http://c.test", "config": None, "credentials": None})
    assert result.exit_code != 0
    assert "No targets given" in result.output


def test_stop_dry_run_lists_jobs_without_sending(monkeypatch):
    monkeypatch.setattr(
        "iris.cli.job._remote_client",
        lambda _ctx: pytest.fail("dry-run must not open a client"),
    )
    result = CliRunner().invoke(
        stop,
        ["--stdin", "--dry-run"],
        input="job_id\n/alice/job\n",
        obj={"controller_url": "http://c.test", "config": None, "credentials": None},
    )
    assert result.exit_code == 0, result.output
    assert "would terminate 1 job(s)" in result.output
    assert "/alice/job" in result.output
