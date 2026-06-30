# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the self-running experiment launcher (``experiments/launch.py``)."""

import dataclasses

import draccus
import pytest
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep

import experiments.launch as launch
from experiments.launch import LaunchConfig, apply_overrides, override_resources


def _handle(runtime_args: dict) -> ArtifactStep:
    """A minimal handle carrying ``runtime_args`` (the build/run callables never fire here)."""
    return ArtifactStep(
        name="test/handle",
        version="dev",
        artifact_type=Artifact,
        run=lambda config: None,
        build_config=lambda ctx: None,
        runtime_args=runtime_args,
    )


# --- override_resources -------------------------------------------------------


def test_override_resources_region_zone_preserves_other_fields():
    base = dataclasses.replace(ResourceConfig.with_tpu("v4-8"), replicas=3)
    out = override_resources(LaunchConfig(region="us-central2", zone="us-central2-b"), base)
    assert out.regions == ("us-central2",)
    assert out.zone == "us-central2-b"
    assert out.replicas == 3
    assert out.device.variant == "v4-8"


def test_override_resources_swaps_tpu_variant_when_vm_count_matches():
    base = ResourceConfig.with_tpu("v5litepod-16")
    out = override_resources(LaunchConfig(tpu_type="v6e-16"), base)
    assert out.device.variant == "v6e-16"


def test_override_resources_rejects_vm_count_mismatch():
    base = ResourceConfig.with_tpu("v5p-8")  # vm_count 1
    with pytest.raises(ValueError, match="vm_count"):
        override_resources(LaunchConfig(tpu_type="v5p-16"), base)  # vm_count 2


def test_override_resources_rejects_flexible_tpu_config():
    base = ResourceConfig.with_tpu(["v5litepod-8", "v6e-8"])  # same vm_count + chips_per_vm
    assert base.device_alternatives
    with pytest.raises(ValueError, match="flexible"):
        override_resources(LaunchConfig(tpu_type="v5litepod-8"), base)


def test_override_resources_rejects_tpu_type_on_non_tpu():
    with pytest.raises(ValueError, match="only applies to TPU"):
        override_resources(LaunchConfig(tpu_type="v4-8"), ResourceConfig.with_cpu())


def test_override_resources_noop_without_overrides_returns_same_object():
    base = ResourceConfig.with_tpu("v4-8")
    assert override_resources(LaunchConfig(), base) is base


# --- apply_overrides (handle level) -------------------------------------------


def test_apply_overrides_rewrites_train_resources_and_preserves_other_runargs():
    handle = _handle({"train_resources": ResourceConfig.with_tpu("v4-8"), "other": 7})
    out = apply_overrides(LaunchConfig(tpu_type="v4-8", region="us-central2"), handle)
    assert out.runtime_args["train_resources"].regions == ("us-central2",)
    assert out.runtime_args["other"] == 7
    assert out.name == handle.name and out.version == handle.version


def test_apply_overrides_noop_without_overrides_returns_same_handle():
    handle = _handle({"train_resources": ResourceConfig.with_tpu("v4-8")})
    assert apply_overrides(LaunchConfig(), handle) is handle


def test_apply_overrides_without_train_resources_warns_and_returns_unchanged(caplog):
    handle = _handle({})
    out = apply_overrides(LaunchConfig(region="us-central2"), handle)
    assert out is handle
    assert "train_resources" in caplog.text


# --- launch dispatch ----------------------------------------------------------


def _forbid_submit(*_args, **_kwargs):
    raise AssertionError("_submit_coordinator_job should not be called")


def test_launch_in_job_runs_body_and_never_submits(monkeypatch):
    monkeypatch.setattr(launch, "get_job_info", lambda: object())
    monkeypatch.setattr(launch, "_submit_coordinator_job", _forbid_submit)
    seen = []
    launch.launch(LaunchConfig(cluster="marin"), lambda config: seen.append(config))
    assert len(seen) == 1


def test_launch_local_runs_body(monkeypatch):
    monkeypatch.setattr(launch, "get_job_info", lambda: None)
    monkeypatch.setattr(launch, "_submit_coordinator_job", _forbid_submit)
    seen = []
    launch.launch(LaunchConfig(cluster=None), lambda config: seen.append(config))
    assert len(seen) == 1


def test_launch_dry_run_against_cluster_stays_local(monkeypatch):
    monkeypatch.setattr(launch, "get_job_info", lambda: None)
    monkeypatch.setattr(launch, "_submit_coordinator_job", _forbid_submit)
    seen = []
    launch.launch(LaunchConfig(cluster="marin", dry_run=True), lambda config: seen.append(config))
    assert len(seen) == 1


def test_launch_laptop_with_cluster_submits_and_skips_body(monkeypatch):
    monkeypatch.setattr(launch, "get_job_info", lambda: None)
    submitted = []

    def fake_submit(config, body):
        submitted.append((config, body))
        return 0

    monkeypatch.setattr(launch, "_submit_coordinator_job", fake_submit)
    ran = []
    with pytest.raises(SystemExit) as exc:
        launch.launch(LaunchConfig(cluster="marin"), lambda config: ran.append(True))
    assert exc.value.code == 0
    assert len(submitted) == 1
    assert ran == []


# --- LaunchConfig surface -----------------------------------------------------


def test_launch_config_subclass_parses_its_own_flags():
    @dataclasses.dataclass
    class WithDevice(LaunchConfig):
        device: str = "cpu"

    parsed = draccus.parse(WithDevice, args=["--cluster", "marin", "--device", "h100x8", "--follow", "true"])
    assert parsed.cluster == "marin"
    assert parsed.device == "h100x8"
    assert parsed.follow is True
