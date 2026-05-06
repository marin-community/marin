# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cli.job — validate_region_zone, executor heuristic, and related CLI validation."""

import click
import pytest
from iris.cli.job import (
    _parse_tpu_alternatives,
    _render_job_summary_text,
    build_job_constraints,
    build_job_summary,
    build_resources,
    build_tpu_alternatives,
    validate_extra_resources,
    validate_region_zone,
)
from iris.cluster.constraints import (
    Constraint,
    WellKnownAttribute,
    infer_preemptible_constraint,
    preemptible_constraint,
    region_constraint,
)
from iris.rpc import config_pb2
from iris.rpc import job_pb2 as _job_pb2


def _make_config_with_zones(zones: list[str]) -> config_pb2.IrisClusterConfig:
    """Build a minimal IrisClusterConfig with scale groups for the given zones."""
    config = config_pb2.IrisClusterConfig()
    for zone in zones:
        region = zone.rsplit("-", 1)[0]
        sg = config.scale_groups[f"sg-{zone}"]
        sg.worker.attributes["zone"] = zone
        sg.worker.attributes["region"] = region
    return config


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
