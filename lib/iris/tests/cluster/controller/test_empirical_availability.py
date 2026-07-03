# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for empirical availability and the availability probe (routing.py).

``empirical_zone_capabilities`` derives the zone -> variant availability map from
LIVE scaling-group state (>0 READY slices, not erroring), not from configuration.
``availability_probe_entries`` converts an unsatisfiable ``availability:<variant>``
constraint into one synthetic accelerator scale-up, so the cluster can DISCOVER
whether that variant can actually be obtained in some region.
"""

from iris.cluster.config import GcpSliceConfig, ScaleGroupConfig, ScaleGroupResources, SliceConfig
from iris.cluster.constraints import (
    ConstraintOp,
    WellKnownAttribute,
    availability_constraint,
    extract_placement_requirements,
    region_constraint,
)
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.autoscaler.routing import (
    availability_probe_entries,
    empirical_zone_capabilities,
)
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.types import AcceleratorType
from iris.rpc import job_pb2
from rigging.timing import Timestamp
from tests.cluster.backends.conftest import make_fake_slice_handle, make_mock_platform

TS = Timestamp.from_ms(1_000_000)


def _group(
    name: str,
    zone: str,
    *,
    variant: str = "v5p-8",
    device_type: AcceleratorType = AcceleratorType.TPU,
    device_count: int = 8,
    ready: int = 0,
) -> ScalingGroup:
    """A ScalingGroup in ``zone`` for ``variant``, optionally with ``ready`` READY slices."""
    config = ScaleGroupConfig(
        name=name,
        buffer_slices=0,
        max_slices=10,
        num_vms=1,
        slice_template=SliceConfig(gcp=GcpSliceConfig(runtime_version="v2-alpha-tpuv5", zone=zone)),
        resources=ScaleGroupResources(
            cpu_millicores=64_000,
            memory_bytes=64 * 1024**3,
            disk_bytes=100 * 1024**3,
            device_type=device_type,
            device_variant=variant,
            device_count=device_count,
        ),
    )
    discovered = [make_fake_slice_handle(f"{name}-{i}", scale_group=name, all_ready=True) for i in range(ready)]
    platform = make_mock_platform(slices_to_discover=discovered)
    group = ScalingGroup(config, platform)
    if discovered:
        group.reconcile()
        for handle in discovered:
            worker_ids = [vm.worker_id for vm in handle.describe().workers]
            group.mark_slice_ready(handle.slice_id, worker_ids, timestamp=TS)
    return group


def _demand(constraints: list) -> DemandEntry:
    return DemandEntry(
        task_ids=("u/j/0",),
        coschedule_group_id=None,
        normalized=extract_placement_requirements(constraints),
        constraints=constraints,
        resources=job_pb2.ResourceSpecProto(),
    )


class TestEmpiricalZoneCapabilities:
    def test_ready_group_makes_variant_available(self):
        caps = empirical_zone_capabilities([_group("g", "us-central1-a", ready=1)], TS)
        assert caps == {"us-central1-a": frozenset({"v5p-8"})}

    def test_group_without_ready_slices_advertises_nothing(self):
        # The core empirical change: a configured-but-never-launched group is silent
        # until a scale-up actually yields a READY slice.
        caps = empirical_zone_capabilities([_group("g", "us-central1-a", ready=0)], TS)
        assert caps == {}

    def test_quota_exceeded_group_excluded_despite_ready_slices(self):
        # ">0 slices allocated AND not erroring": a group that has live slices but is
        # now quota-blocked is not advertised as available.
        group = _group("g", "us-central1-a", ready=1)
        group.record_quota_exceeded("out of quota", timestamp=TS)
        assert empirical_zone_capabilities([group], TS) == {}

    def test_variant_rolls_up_to_every_zone_in_region(self):
        # Availability is a regional question: a live v5p-8 slice in us-central1-a
        # makes v5p-8 available to a sibling group in us-central1-b too.
        live = _group("live", "us-central1-a", ready=1)
        sibling = _group("sibling", "us-central1-b", ready=0)
        caps = empirical_zone_capabilities([live, sibling], TS)
        assert caps == {
            "us-central1-a": frozenset({"v5p-8"}),
            "us-central1-b": frozenset({"v5p-8"}),
        }

    def test_other_region_unaffected(self):
        live = _group("live", "us-central1-a", ready=1)
        cold = _group("cold", "us-east5-b", ready=0)
        caps = empirical_zone_capabilities([live, cold], TS)
        assert "us-east5-b" not in caps

    def test_variants_are_lowercased(self):
        caps = empirical_zone_capabilities([_group("g", "us-east5-b", variant="V6E-8", ready=1)], TS)
        assert caps == {"us-east5-b": frozenset({"v6e-8"})}


class TestAvailabilityProbeEntries:
    def test_probe_emitted_for_unavailable_demanded_variant(self):
        groups = [_group("g", "us-central1-a", variant="v5p-8")]
        demand = [_demand([availability_constraint("v5p-8")])]
        probes = availability_probe_entries(groups, demand, available_variants=frozenset())

        assert len(probes) == 1
        probe = probes[0]
        assert probe.task_ids[0].startswith("__availability_probe__")
        assert probe.coschedule_group_id is None
        # Routed by device variant (not by availability, which would be circular).
        assert len(probe.constraints) == 1
        constraint = probe.constraints[0]
        assert (constraint.key, constraint.op) == (WellKnownAttribute.DEVICE_VARIANT, ConstraintOp.EQ)
        assert [v.value for v in constraint.values] == ["v5p-8"]
        assert probe.resources.device.HasField("tpu")
        assert probe.resources.device.tpu.variant == "v5p-8"

    def test_no_probe_when_variant_already_available(self):
        groups = [_group("g", "us-central1-a", variant="v5p-8")]
        demand = [_demand([availability_constraint("v5p-8")])]
        assert availability_probe_entries(groups, demand, available_variants=frozenset({"v5p-8"})) == []

    def test_no_probe_without_a_matching_group(self):
        # Demand for a variant no configured group provides cannot be probed.
        groups = [_group("g", "us-central1-a", variant="v4-8")]
        demand = [_demand([availability_constraint("v5p-8")])]
        assert availability_probe_entries(groups, demand, available_variants=frozenset()) == []

    def test_no_probe_without_availability_demand(self):
        groups = [_group("g", "us-central1-a", variant="v5p-8")]
        demand = [_demand([region_constraint(["us-central1"])])]
        assert availability_probe_entries(groups, demand, available_variants=frozenset()) == []

    def test_one_probe_per_variant_across_many_waiting_jobs(self):
        groups = [_group("g", "us-central1-a", variant="v5p-8")]
        demand = [_demand([availability_constraint("v5p-8")]) for _ in range(5)]
        probes = availability_probe_entries(groups, demand, available_variants=frozenset())
        assert len(probes) == 1

    def test_gpu_probe_uses_gpu_device(self):
        groups = [_group("g", "us-east5-b", variant="h100", device_type=AcceleratorType.GPU)]
        demand = [_demand([availability_constraint("h100")])]
        probes = availability_probe_entries(groups, demand, available_variants=frozenset())

        assert len(probes) == 1
        assert probes[0].resources.device.HasField("gpu")
        assert probes[0].resources.device.gpu.variant == "h100"
