# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the self-running experiment launcher (``experiments/launch.py``)."""

import dataclasses
import os

import draccus
import pytest
from fray.current_client import current_client
from fray.local_backend import LocalClient
from fray.types import GpuConfig, ResourceConfig
from marin.execution.executor import ExecutorStep

import experiments.launch as launch
from experiments.launch import LaunchConfig, _ensure_storage_prefix, launch_session, override_resources


def test_launch_config_parses_embedded_executor_flags():
    cfg = draccus.parse(
        LaunchConfig,
        args=["--cluster=marin", "--tpu_type=v4-8", "--region=us-central2", "--executor.dry_run=True"],
    )
    assert cfg.cluster == "marin"
    assert cfg.tpu_type == "v4-8"
    assert cfg.region == "us-central2"
    assert cfg.executor.dry_run is True


def test_override_resources_region_zone_preserves_other_fields():
    base = ResourceConfig.with_tpu("v5p-8", slice_count=2, regions=["eu-west4"], preemptible=False)
    out = override_resources(base, LaunchConfig(region="us-central2", zone="us-central2-b"))
    assert out.regions == ("us-central2",)
    assert out.zone == "us-central2-b"
    # Untouched scheduling fields survive.
    assert out.replicas == base.replicas
    assert out.preemptible is False
    assert out.device.variant == "v5p-8"


def test_override_resources_swaps_tpu_variant_when_vm_count_matches():
    # v5p-8 and v4-8 are both single-VM, so the replica count stays valid.
    base = ResourceConfig.with_tpu("v5p-8", preemptible=False)
    out = override_resources(base, LaunchConfig(tpu_type="v4-8"))
    assert out.device.variant == "v4-8"
    assert out.replicas == base.replicas
    assert out.preemptible is False


def test_override_resources_rejects_vm_count_mismatch():
    # v5p-16 spans two VMs; swapping a single-VM default would corrupt replicas.
    base = ResourceConfig.with_tpu("v5p-8")
    with pytest.raises(ValueError, match="vm_count"):
        override_resources(base, LaunchConfig(tpu_type="v5p-16"))


def test_override_resources_rejects_flexible_tpu_config():
    base = ResourceConfig.with_tpu(["v5p-8", "v4-8"])
    with pytest.raises(ValueError, match="flexible"):
        override_resources(base, LaunchConfig(tpu_type="v4-8"))


def test_override_resources_rejects_tpu_type_on_non_tpu():
    base = ResourceConfig.with_gpu("H100", count=8)
    assert isinstance(base.device, GpuConfig)
    with pytest.raises(ValueError, match="TPU"):
        override_resources(base, LaunchConfig(tpu_type="v4-8"))


def test_override_resources_noop_without_overrides():
    base = ResourceConfig.with_tpu("v5p-8")
    assert override_resources(base, LaunchConfig()) is base


def test_ensure_storage_prefix_requires_regional_gcs(monkeypatch):
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    with pytest.raises(ValueError, match="MARIN_PREFIX"):
        _ensure_storage_prefix("marin")

    monkeypatch.setenv("MARIN_PREFIX", "/tmp/marin")
    with pytest.raises(ValueError, match="MARIN_PREFIX"):
        _ensure_storage_prefix("marin")

    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central2")
    _ensure_storage_prefix("marin")  # does not raise


@dataclasses.dataclass(frozen=True)
class _PodConfigStub:
    resources: ResourceConfig
    note: str = "x"


def test_apply_overrides_updates_both_step_and_config_resources():
    base = ResourceConfig.with_tpu("v5p-8")
    step = ExecutorStep(
        name="train/x",
        fn=lambda p: None,
        config=_PodConfigStub(resources=base),
        resources=base,
    )
    out = launch._apply_overrides_to_step(step, LaunchConfig(tpu_type="v4-8", region="us-central2"))
    # Both the scheduling resources and the embedded config resources are overridden in sync.
    assert out.resources.device.variant == "v4-8"
    assert out.resources.regions == ("us-central2",)
    assert out.config.resources.device.variant == "v4-8"
    assert out.config.resources.regions == ("us-central2",)
    # Non-resource config fields survive.
    assert out.config.note == "x"


def test_apply_overrides_leaves_inline_cpu_step_untouched():
    # resources=None (the default) => inline step; nothing to override even with flags set.
    step = ExecutorStep(name="data/x", fn=lambda p: None, config=_PodConfigStub(resources=ResourceConfig.with_cpu()))
    assert step.resources is None
    assert launch._apply_overrides_to_step(step, LaunchConfig(region="us-central2")) is step


def test_launch_session_hard_fails_when_cluster_passed_inside_job(monkeypatch):
    monkeypatch.setattr(launch, "get_job_info", lambda: object())
    with pytest.raises(RuntimeError, match="inside an Iris job"):
        with launch_session(LaunchConfig(cluster="marin")):
            pass


def test_launch_session_warns_and_continues_for_legacy_two_hop(monkeypatch, caplog):
    monkeypatch.setattr(launch, "get_job_info", lambda: object())
    entered = False
    with caplog.at_level("WARNING"):
        with launch_session(LaunchConfig(cluster=None)):
            entered = True
    assert entered
    assert any("no longer need" in r.message for r in caplog.records)


def test_launch_session_local_when_no_cluster(monkeypatch):
    monkeypatch.setattr(launch, "get_job_info", lambda: None)
    with launch_session(LaunchConfig(cluster=None)):
        # No client was hoisted: current_client falls back to LocalClient.
        assert isinstance(current_client(), LocalClient)


def test_launch_session_local_does_not_require_marin_prefix(monkeypatch):
    monkeypatch.setattr(launch, "get_job_info", lambda: None)
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    # Local mode must not trip the storage-prefix guard.
    with launch_session(LaunchConfig(cluster=None, local=True)):
        pass
    assert "MARIN_PREFIX" not in os.environ
