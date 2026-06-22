# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the self-running experiment launcher (``experiments/launch.py``)."""

import contextlib
import dataclasses

import draccus
import pytest
from fray.current_client import current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from iris.cluster.constraints import region_constraint, zone_constraint
from marin.execution.executor import ExecutorMainConfig, ExecutorStep
from marin.execution.types import VersionedValue, versioned

import experiments.launch as launch
from experiments.launch import LaunchConfig, override_resources


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
    base = ResourceConfig.with_tpu("v5p-8")
    out = override_resources(base, LaunchConfig(tpu_type="v4-8"))
    assert out.device.variant == "v4-8"
    assert out.replicas == base.replicas


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
    with pytest.raises(ValueError, match="TPU"):
        override_resources(base, LaunchConfig(tpu_type="v4-8"))


def test_override_resources_noop_without_overrides():
    base = ResourceConfig.with_tpu("v5p-8")
    assert override_resources(base, LaunchConfig()) is base


@dataclasses.dataclass(frozen=True)
class _PodConfigStub:
    resources: ResourceConfig | VersionedValue[ResourceConfig]
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


def test_apply_overrides_updates_versioned_config_resources():
    # grug executor steps leave ExecutorStep.resources unset and stash the real
    # TPU config in config.resources wrapped in versioned(...); run_grug submits
    # from config.resources, so the override must reach it and keep the wrapper.
    base = ResourceConfig.with_tpu("v5p-8")
    step = ExecutorStep(name="train/x", fn=lambda p: None, config=_PodConfigStub(resources=versioned(base)))
    out = launch._apply_overrides_to_step(step, LaunchConfig(tpu_type="v4-8", region="us-central2"))
    assert isinstance(out.config.resources, VersionedValue)
    assert out.config.resources.value.device.variant == "v4-8"
    assert out.config.resources.value.regions == ("us-central2",)


def test_apply_overrides_leaves_inline_cpu_step_untouched():
    # A CPU data-prep step keeps its cached identity: even with config.resources set
    # and --region passed, a CPU device is not an accelerator target, so it's skipped.
    step = ExecutorStep(name="data/x", fn=lambda p: None, config=_PodConfigStub(resources=ResourceConfig.with_cpu()))
    assert launch._apply_overrides_to_step(step, LaunchConfig(region="us-central2")) is step


def _must_not_bootstrap(*args, **kwargs):
    raise AssertionError("must not bootstrap a coordinator")


def _noop_body() -> None:
    pass


@pytest.mark.parametrize("cluster", [None, "marin"])
def test_launch_in_job_runs_body_and_never_bootstraps(monkeypatch, cluster):
    # Already inside an Iris job (the coordinator): the body runs *here* against
    # the in-cluster client; an inert --cluster must not trigger another bootstrap.
    monkeypatch.setattr(launch, "get_job_info", lambda: object())
    monkeypatch.setattr(launch, "_submit_coordinator_job", _must_not_bootstrap)
    ran = []
    launch.launch(LaunchConfig(cluster=cluster), lambda *a, **k: ran.append((a, k)), 1, x=2)
    assert ran == [((1,), {"x": 2})]


def test_launch_local_runs_body_with_local_client(monkeypatch):
    # No --cluster: the body runs in-process and current_client() is LocalClient.
    monkeypatch.setattr(launch, "get_job_info", lambda: None)
    seen = {}
    launch.launch(LaunchConfig(cluster=None), lambda: seen.update(client=current_client()))
    assert isinstance(seen["client"], LocalClient)


def test_launch_dry_run_against_cluster_stays_local(monkeypatch):
    # A dry run against a cluster stays in-process — no coordinator job.
    monkeypatch.setattr(launch, "get_job_info", lambda: None)
    monkeypatch.setattr(launch, "_submit_coordinator_job", _must_not_bootstrap)
    ran = []
    cfg = LaunchConfig(cluster="marin", executor=ExecutorMainConfig(dry_run=True))
    launch.launch(cfg, lambda: ran.append(True))
    assert ran == [True]


def test_launch_laptop_with_cluster_bootstraps_and_skips_body(monkeypatch):
    # Laptop + --cluster: launch ships the body (and its args/kwargs) to a
    # coordinator job and exits with its status; the body never runs locally.
    monkeypatch.setattr(launch, "get_job_info", lambda: None)
    captured = {}

    def _fake_bootstrap(config, body, args, kwargs):
        captured.update(cluster=config.cluster, args=args, kwargs=kwargs)
        return 0

    monkeypatch.setattr(launch, "_submit_coordinator_job", _fake_bootstrap)
    ran = []
    with pytest.raises(SystemExit) as exc:
        launch.launch(LaunchConfig(cluster="marin"), lambda *a, **k: ran.append(True), 7, x=9)
    assert exc.value.code == 0
    assert captured == {"cluster": "marin", "args": (7,), "kwargs": {"x": 9}}
    assert ran == []


@pytest.mark.parametrize(("detach", "expect_stream"), [(False, True), (True, False)])
def test_submit_coordinator_job_detach_controls_streaming(monkeypatch, detach, expect_stream):
    # --detach submits the coordinator without streaming; the default streams.
    # Pins the CPU-only coordinator extras and exercises the real
    # Entrypoint.from_callable packing of the body.
    class _FakeJob:
        job_id = "job-1"

    submitted = {}

    class _FakeClient:
        def submit(self, *, entrypoint, name, resources, environment, constraints):
            submitted.update(name=name, extras=environment.extras, constraints=constraints)
            return _FakeJob()

    @contextlib.contextmanager
    def _fake_connect(cluster, *, workspace):
        yield _FakeClient()

    streamed = []
    monkeypatch.setattr(launch, "connect_to_cluster", _fake_connect)
    monkeypatch.setattr(launch, "stream_until_complete", lambda client, job: streamed.append(job) or 0)
    monkeypatch.setattr(launch, "load_env_vars", lambda flags: {})
    monkeypatch.setattr(launch, "add_standard_env_vars", lambda env: env)

    rc = launch._submit_coordinator_job(LaunchConfig(cluster="marin", detach=detach), _noop_body, (), {})
    assert rc == 0
    assert submitted["extras"] == ["cpu"]
    # No --region/--zone, so the coordinator is unconstrained.
    assert submitted["constraints"] is None
    assert bool(streamed) is expect_stream


def test_coordinator_constraints_pin_region_and_zone():
    # --region/--zone must pin the coordinator (which bakes the executor's output
    # paths) to the same place --region sends the training. Unset = no constraint.
    assert launch._coordinator_constraints(LaunchConfig(cluster="marin")) is None
    assert launch._coordinator_constraints(LaunchConfig(cluster="marin", region="us-central2")) == [
        region_constraint(["us-central2"])
    ]
    assert launch._coordinator_constraints(LaunchConfig(cluster="marin", zone="us-central2-b")) == [
        zone_constraint("us-central2-b")
    ]
    assert launch._coordinator_constraints(
        LaunchConfig(cluster="marin", region="us-central2", zone="us-central2-b")
    ) == [region_constraint(["us-central2"]), zone_constraint("us-central2-b")]
