# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for iris job CLI helpers."""

import sys

import click
import pytest
from click.testing import CliRunner
from iris.cli.job import (
    build_resources,
    load_env_vars,
    parse_gpu_spec,
    reserve_spec_to_availability,
    run_iris_job,
)
from iris.cli.job import run as run_cmd
from iris.cluster.config import IrisConfig
from iris.cluster.constraints import ConstraintOp, WellKnownAttribute, availability_key
from iris.cluster.types import JobName
from iris.rpc import job_pb2


def test_load_env_vars_single_key():
    """Test env var with no value (empty string)."""
    result = load_env_vars([["KEY_ONLY"]])
    assert result["KEY_ONLY"] == ""


def test_load_env_vars_invalid_key():
    """Test error on key with = sign."""
    with pytest.raises(ValueError, match="cannot contain '='"):
        load_env_vars([["KEY=VALUE"]])


def test_iris_config_missing_file(tmp_path):
    """Test error on missing config file."""
    with pytest.raises(FileNotFoundError):
        IrisConfig.load(tmp_path / "nonexistent.yaml")


def test_iris_config_empty_file(tmp_path):
    """Test error on empty config file."""
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("")
    with pytest.raises(ValueError, match="Config file is empty"):
        IrisConfig.load(bad_config)


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("H100x8", ("H100", 8)),
        ("4", ("", 4)),
        ("A100", ("A100", 1)),
        ("rtx4090", ("RTX4090", 1)),
        ("rtx4090x2", ("RTX4090", 2)),
        ("H100", ("H100", 1)),
    ],
)
def test_parse_gpu_spec(spec, expected):
    assert parse_gpu_spec(spec) == expected


@pytest.mark.parametrize("spec", ["0", ""])
def test_parse_gpu_spec_rejects_invalid(spec):
    with pytest.raises(ValueError):
        parse_gpu_spec(spec)


def test_reserve_spec_to_availability_tpu():
    """A TPU variant yields a soft availability:<variant> EXISTS constraint."""
    constraint = reserve_spec_to_availability("v5litepod-16")
    assert constraint.key == availability_key("v5litepod-16")
    assert constraint.op == ConstraintOp.EXISTS
    assert constraint.is_soft


def test_reserve_spec_to_availability_gpu():
    """A GPU spec keys on the GPU variant (count suffix ignored)."""
    constraint = reserve_spec_to_availability("H100x8")
    assert constraint.key == availability_key("H100")
    assert constraint.op == ConstraintOp.EXISTS
    assert constraint.is_soft


def test_reserve_spec_to_availability_count_prefix_ignored():
    """A leading COUNT: prefix is accepted and produces the same key."""
    assert reserve_spec_to_availability("4:H100x8").key == reserve_spec_to_availability("H100x8").key


def test_reserve_spec_to_availability_rejects_non_accelerator():
    with pytest.raises(click.UsageError):
        reserve_spec_to_availability("4")


def test_build_resources_gpu():
    """Test GPU spec parsing in build_resources."""
    spec = build_resources(tpu=None, gpu="H100x8")
    assert spec.device.HasField("gpu")
    assert spec.device.gpu.variant == "H100"
    assert spec.device.gpu.count == 8

    # Bare count defaults to empty variant
    spec = build_resources(tpu=None, gpu="4")
    assert spec.device.gpu.variant == ""
    assert spec.device.gpu.count == 4

    # Bare variant defaults to count=1
    spec = build_resources(tpu=None, gpu="A100")
    assert spec.device.gpu.variant == "A100"
    assert spec.device.gpu.count == 1


def test_run_iris_job_adds_zone_constraint(monkeypatch):
    """run_iris_job forwards a zone placement constraint."""
    captured: dict[str, object] = {}

    def _fake_submit_and_wait_job(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("iris.cli.job._submit_and_wait_job", _fake_submit_and_wait_job)

    exit_code = run_iris_job(
        controller_url="http://controller:10000",
        command=[sys.executable, "-c", "print('ok')"],
        env_vars={},
        wait=False,
        zone="us-central2-b",
    )

    assert exit_code == 0
    constraints = captured["constraints"]
    assert constraints is not None

    zone_constraints = [c for c in constraints if c.key == WellKnownAttribute.ZONE]
    assert len(zone_constraints) == 1
    assert zone_constraints[0].op == ConstraintOp.EQ
    assert zone_constraints[0].values[0].value == "us-central2-b"


def test_run_iris_job_passes_reserve_as_availability_constraint(monkeypatch):
    """run_iris_job forwards --reserve as a soft availability constraint."""
    captured: dict[str, object] = {}

    def _fake_submit_and_wait_job(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("iris.cli.job._submit_and_wait_job", _fake_submit_and_wait_job)

    exit_code = run_iris_job(
        controller_url="http://controller:10000",
        command=[sys.executable, "-c", "print('ok')"],
        env_vars={},
        wait=False,
        reserve=("4:H100x8",),
    )

    assert exit_code == 0
    constraints = captured["constraints"]
    assert constraints is not None
    availability = [c for c in constraints if c.key == availability_key("H100")]
    assert len(availability) == 1
    assert availability[0].op == ConstraintOp.EXISTS
    assert availability[0].is_soft


def test_run_iris_job_adds_region_and_zone_constraints(monkeypatch):
    """run_iris_job combines region and zone constraints when both are set."""
    captured: dict[str, object] = {}

    def _fake_submit_and_wait_job(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("iris.cli.job._submit_and_wait_job", _fake_submit_and_wait_job)

    exit_code = run_iris_job(
        controller_url="http://controller:10000",
        command=[sys.executable, "-c", "print('ok')"],
        env_vars={},
        wait=False,
        regions=("us-central2",),
        zone="us-central2-b",
    )

    assert exit_code == 0
    constraints = captured["constraints"]
    assert constraints is not None

    region_constraints = [c for c in constraints if c.key == WellKnownAttribute.REGION]
    assert len(region_constraints) == 1
    assert region_constraints[0].op == ConstraintOp.EQ
    assert region_constraints[0].values[0].value == "us-central2"

    zone_constraints = [c for c in constraints if c.key == WellKnownAttribute.ZONE]
    assert len(zone_constraints) == 1
    assert zone_constraints[0].op == ConstraintOp.EQ
    assert zone_constraints[0].values[0].value == "us-central2-b"


def test_run_iris_job_passes_priority_band(monkeypatch):
    """run_iris_job converts a priority name to its proto value."""

    captured: dict[str, object] = {}

    def _fake_submit_and_wait_job(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("iris.cli.job._submit_and_wait_job", _fake_submit_and_wait_job)

    exit_code = run_iris_job(
        controller_url="http://controller:10000",
        command=[sys.executable, "-c", "print('ok')"],
        env_vars={},
        wait=False,
        priority="batch",
    )

    assert exit_code == 0
    assert captured["priority_band"] == job_pb2.PRIORITY_BAND_BATCH


def test_run_iris_job_default_priority_unspecified(monkeypatch):
    """run_iris_job defaults to PRIORITY_BAND_UNSPECIFIED when --priority is omitted."""

    captured: dict[str, object] = {}

    def _fake_submit_and_wait_job(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("iris.cli.job._submit_and_wait_job", _fake_submit_and_wait_job)

    exit_code = run_iris_job(
        controller_url="http://controller:10000",
        command=[sys.executable, "-c", "print('ok')"],
        env_vars={},
        wait=False,
    )

    assert exit_code == 0
    assert captured["priority_band"] == job_pb2.PRIORITY_BAND_UNSPECIFIED


def test_no_wait_prints_job_id(monkeypatch):
    """--no-wait prints the job ID to stdout."""

    class FakeJob:
        job_id = JobName.from_wire("/test-user/test-job")

    class FakeClient:
        def submit(self, **kwargs):
            return FakeJob()

    monkeypatch.setattr("iris.cli.job.IrisClient.remote", lambda *a, **kw: FakeClient())

    runner = CliRunner()
    result = runner.invoke(
        run_cmd,
        ["--no-wait", "--", "echo", "hi"],
        catch_exceptions=False,
        obj={"controller_url": "http://fake:10000"},
    )
    assert result.exit_code == 0
    assert result.output.strip() == "/test-user/test-job"
