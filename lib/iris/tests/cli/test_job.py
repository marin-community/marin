# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Iris Job submission validation and placement policy."""

import pytest
from click.testing import CliRunner
from iris.cli.job import (
    build_job_constraints,
    build_resources,
    run,
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


# ---------------------------------------------------------------------------
# Executor heuristic tests (mirrors the logic in run_iris_job)
# ---------------------------------------------------------------------------


def test_executor_heuristic_small_cpu_job_gets_non_preemptible():
    resources = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB")
    replicas = 1
    constraints: list[Constraint] = []

    preemptible = infer_preemptible_constraint(resources, replicas, constraints)
    assert preemptible is not None
    assert preemptible.key == WellKnownAttribute.PREEMPTIBLE
    assert preemptible.values[0].value == "false"


def test_executor_heuristic_skipped_for_gpu_job():
    resources = build_resources(tpu=None, gpu="H100", cpu=0.5, memory="1GB", disk="5GB")
    replicas = 1
    constraints: list[Constraint] = []

    preemptible = infer_preemptible_constraint(resources, replicas, constraints)
    assert preemptible is None


def test_executor_heuristic_skipped_for_large_cpu_job():
    resources = build_resources(tpu=None, gpu=None, cpu=4.0, memory="16GB", disk="5GB")
    replicas = 1
    constraints: list[Constraint] = []

    preemptible = infer_preemptible_constraint(resources, replicas, constraints)
    assert preemptible is None


def test_executor_heuristic_skipped_when_user_sets_preemptible():
    resources = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB")
    replicas = 1
    constraints: list[Constraint] = [preemptible_constraint(True)]

    preemptible = infer_preemptible_constraint(resources, replicas, constraints)
    assert preemptible is None


def test_executor_heuristic_with_region_constraint():
    resources = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB")
    replicas = 1
    constraints: list[Constraint] = [region_constraint(["us-central2"])]

    preemptible = infer_preemptible_constraint(resources, replicas, constraints)
    assert preemptible is not None
    assert preemptible.values[0].value == "false"


# ---------------------------------------------------------------------------
# build_job_constraints — --preemptible / --no-preemptible wiring (#4540)
# ---------------------------------------------------------------------------


def _preemptible_values(constraints: list[Constraint]) -> list[str]:
    return [c.values[0].value for c in constraints if c.key == WellKnownAttribute.PREEMPTIBLE]


def test_build_job_constraints_preemptible_true_emits_true_constraint():
    """--preemptible forces a preemptible=true constraint and bypasses the heuristic."""
    resources = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB")

    constraints = build_job_constraints(resources, tpu_variants=[], replicas=1, preemptible=True)

    assert _preemptible_values(constraints) == ["true"]


def test_build_job_constraints_preemptible_false_emits_false_constraint():
    """--no-preemptible forces a preemptible=false constraint even for non-executor jobs."""
    resources = build_resources(tpu=None, gpu=None, cpu=4.0, memory="16GB", disk="5GB")

    constraints = build_job_constraints(resources, tpu_variants=[], replicas=1, preemptible=False)

    assert _preemptible_values(constraints) == ["false"]


def test_build_job_constraints_preemptible_none_runs_heuristic():
    """Default (None) preserves the executor heuristic on small CPU jobs."""
    resources = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB")

    constraints = build_job_constraints(resources, tpu_variants=[], replicas=1, preemptible=None)

    assert _preemptible_values(constraints) == ["false"]


def test_build_job_constraints_preemptible_true_overrides_heuristic():
    """Small CPU jobs normally auto-tag non-preemptible; --preemptible wins."""
    resources = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB")

    constraints = build_job_constraints(resources, tpu_variants=[], replicas=1, preemptible=True)

    # Exactly one preemptible constraint, and it reflects the user's choice.
    assert _preemptible_values(constraints) == ["true"]


def test_build_job_constraints_target_cluster_appends_cluster_pin():
    """--target-cluster appends exactly one cluster EQ constraint naming the peer."""
    resources = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB")

    constraints = build_job_constraints(resources, tpu_variants=[], replicas=1, target_cluster="peer-cluster")

    cluster_constraints = [c for c in constraints if c.key == CLUSTER_CONSTRAINT_KEY]
    assert len(cluster_constraints) == 1
    pin = cluster_constraints[0]
    assert pin.op == ConstraintOp.EQ
    assert pin.values[0].value == "peer-cluster"


def test_build_job_constraints_no_target_cluster_omits_cluster_pin():
    """Omitting --target-cluster appends no cluster constraint."""
    resources = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB")

    constraints = build_job_constraints(resources, tpu_variants=[], replicas=1, target_cluster=None)

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
    assert captured["entrypoint"].command == ("python", "train.py")


# --tpu multi-variant parsing
# ---------------------------------------------------------------------------


def test_tpu_multi_variant_parsing(recorded_job_submissions):
    result = _run_cli(
        ["--enable-extra-resources", "--tpu", " v6e-4 , v5litepod-4 , v5p-8 "],
    )
    assert result.exit_code == 0, result.output
    submission = recorded_job_submissions[0]
    assert submission["resources"].device.variant == "v6e-4"
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
