# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the fray Iris backend.

Tests type conversions and handle serialization without requiring an Iris cluster.
Integration tests that need a running cluster are marked with @pytest.mark.iris.
"""

import pickle
from unittest.mock import MagicMock

import pytest
from fray.iris_backend import (
    FrayIrisClient,
    IrisActorHandle,
    convert_constraints,
    resolve_coscheduling,
    wrap_multiprocess,
)
from fray.types import (
    ANY_REGION,
    CpuConfig,
    Entrypoint,
    GpuConfig,
    JobRequest,
    ResourceConfig,
    TpuConfig,
)
from iris.cluster.constraints import ConstraintOp
from iris.cluster.types import Entrypoint as IrisEntrypoint
from iris.cluster.types import ResourceSpec, gpu_device


class TestConvertConstraints:
    def test_preemptible_true_produces_no_constraints(self):
        resources = ResourceConfig(preemptible=True)
        constraints = convert_constraints(resources)
        assert constraints == []

    def test_preemptible_false_adds_constraint(self):
        resources = ResourceConfig(preemptible=False)
        constraints = convert_constraints(resources)
        assert len(constraints) == 1
        c = constraints[0]
        assert c.key == "preemptible"
        assert c.values[0].value == "false"

    def test_single_region_produces_eq_constraint(self):
        resources = ResourceConfig(regions=["us-central1"])
        constraints = convert_constraints(resources)
        region_constraints = [c for c in constraints if c.key == "region"]
        assert len(region_constraints) == 1
        c = region_constraints[0]

        assert c.op == ConstraintOp.EQ
        assert c.values[0].value == "us-central1"

    def test_multiple_regions_produce_in_constraint(self):
        resources = ResourceConfig(regions=["us-central1", "us-central2"])
        constraints = convert_constraints(resources)
        region_constraints = [c for c in constraints if c.key == "region"]
        assert len(region_constraints) == 1
        c = region_constraints[0]

        assert c.op == ConstraintOp.IN
        assert tuple(v.value for v in c.values) == ("us-central1", "us-central2")

    def test_regions_unset_produces_no_region_constraint(self):
        """UNSET (regions=None, the default) emits no region constraint — the child
        inherits the parent's region at submit time."""
        resources = ResourceConfig(regions=None)
        constraints = convert_constraints(resources)
        assert [c for c in constraints if c.key == "region"] == []

    def test_any_region_produces_single_exists_constraint(self):
        """ANY ([ANY_REGION]) emits exactly one region-EXISTS marker."""
        resources = ResourceConfig(regions=[ANY_REGION])
        constraints = convert_constraints(resources)
        region_constraints = [c for c in constraints if c.key == "region"]
        assert len(region_constraints) == 1
        assert region_constraints[0].op == ConstraintOp.EXISTS
        assert region_constraints[0].values == ()

    def test_any_region_mixed_with_specific_region_raises(self):
        """ANY_REGION alongside a concrete region is contradictory and rejected."""
        resources = ResourceConfig(regions=[ANY_REGION, "us-central1"])
        with pytest.raises(ValueError, match="ANY_REGION cannot be combined"):
            convert_constraints(resources)

    def test_zone_produces_eq_constraint(self):
        resources = ResourceConfig(zone="us-east1-d")
        constraints = convert_constraints(resources)
        zone_constraints = [c for c in constraints if c.key == "zone"]
        assert len(zone_constraints) == 1
        c = zone_constraints[0]

        assert c.op == ConstraintOp.EQ
        assert c.values[0].value == "us-east1-d"


class TestConvertConstraintsDeviceAlternatives:
    def test_no_alternatives_produces_no_device_constraint(self):
        resources = ResourceConfig.with_tpu("v5p-8")
        constraints = convert_constraints(resources)
        device_constraints = [c for c in constraints if c.key == "device-variant"]
        assert device_constraints == []

    def test_alternatives_produce_in_constraint(self):
        resources = ResourceConfig.with_tpu(["v4-8", "v5p-8"])
        constraints = convert_constraints(resources)
        device_constraints = [c for c in constraints if c.key == "device-variant"]
        assert len(device_constraints) == 1
        c = device_constraints[0]

        assert c.op == ConstraintOp.IN
        assert {v.value for v in c.values} == {"v4-8", "v5p-8"}


class TestIrisActorHandlePickle:
    def test_pickle_roundtrip_preserves_name(self):
        handle = IrisActorHandle("my-actor")
        data = pickle.dumps(handle)
        restored = pickle.loads(data)
        assert restored._endpoint_name == "my-actor"
        assert restored._client is None

    def test_pickle_drops_client(self):
        """Client is transient state — pickle should not carry it."""
        handle = IrisActorHandle("my-actor")
        # Manually set client to simulate resolved state
        handle._client = "fake-client"
        data = pickle.dumps(handle)
        restored = pickle.loads(data)
        assert restored._client is None


class TestResourceConfigScale:
    def test_scale_with_uniform_factor_scales_all_dimensions(self):
        scaled = ResourceConfig(cpu=1, ram="4g", disk="2g").scale(2)
        assert scaled.cpu == 2
        assert scaled.ram == "8g"
        assert scaled.disk == "4g"

    def test_scale_with_fractional_factor_scales_all_dimensions_down(self):
        scaled = ResourceConfig(cpu=2, ram="8g", disk="4g").scale(0.5)
        assert scaled.cpu == 1
        assert scaled.ram == "4g"
        assert scaled.disk == "2g"

    def test_scale_with_individual_factors_scales_specified_dimensions(self):
        base = ResourceConfig(cpu=1, ram="4g", disk="2g")
        scaled = base.scale(cpu=3, ram=3, disk=2.5)
        assert scaled.cpu == 3
        assert scaled.ram == "12g"
        assert scaled.disk == "5g"

    def test_scale_with_omitted_kwargs_leaves_unspecified_dimensions_unchanged(self):
        scaled = ResourceConfig(cpu=1, ram="10g", disk="2g").scale(cpu=2, ram=2.4)
        assert scaled.cpu == 2
        assert scaled.ram == "24g"
        assert scaled.disk == "2g"

    def test_scale_with_factor_preserves_non_size_fields(self):
        base = ResourceConfig(cpu=1, ram="4g", disk="2g", preemptible=False, image="custom")
        scaled = base.scale(2)
        assert scaled.preemptible is False
        assert scaled.image == "custom"

    def test_scale_with_mixed_factor_and_kwargs_raises_value_error(self):
        with pytest.raises(ValueError, match="either a single factor"):
            ResourceConfig(cpu=1, ram="4g").scale(2, cpu=3)


class TestImagePlumbing:
    def test_resource_config_image_default_is_none(self):
        rc = ResourceConfig()
        assert rc.image is None

    def test_resource_config_image_set(self):
        rc = ResourceConfig(image="custom/swetrace:dev")
        assert rc.image == "custom/swetrace:dev"

    def test_create_actor_group_passes_task_image_to_iris(self):
        """resources.image must reach the underlying iris.submit() call as task_image."""
        fake_iris = MagicMock()
        fake_iris.submit.return_value = MagicMock(job_id="job-123")
        client = FrayIrisClient.from_iris_client(fake_iris)

        class _DummyActor:
            pass

        client.create_actor_group(
            _DummyActor,
            name="dummy",
            count=2,
            resources=ResourceConfig(cpu=2, ram="4g", image="custom/swetrace:dev"),
        )

        kwargs = fake_iris.submit.call_args.kwargs
        assert kwargs["task_image"] == "custom/swetrace:dev"
        assert kwargs["replicas"] == 2

    def test_create_actor_group_default_image_is_none(self):
        """When ResourceConfig.image is unset, task_image flows through as None."""
        fake_iris = MagicMock()
        fake_iris.submit.return_value = MagicMock(job_id="job-123")
        client = FrayIrisClient.from_iris_client(fake_iris)

        class _DummyActor:
            pass

        client.create_actor_group(_DummyActor, name="dummy", count=1)

        kwargs = fake_iris.submit.call_args.kwargs
        assert kwargs["task_image"] is None

    def test_submit_job_passes_task_image_to_iris(self):
        """resources.image on a top-level job request reaches iris.submit()."""
        fake_iris = MagicMock()
        fake_iris.submit.return_value = MagicMock(job_id="job-456")
        client = FrayIrisClient.from_iris_client(fake_iris)

        def _noop():
            return None

        request = JobRequest(
            name="test-job",
            entrypoint=Entrypoint.from_callable(_noop),
            resources=ResourceConfig(cpu=1, ram="2g", image="custom/swetrace:dev"),
        )
        client.submit(request)

        kwargs = fake_iris.submit.call_args.kwargs
        assert kwargs["task_image"] == "custom/swetrace:dev"


class TestActorGroupEnvironment:
    """Verify create_actor_group passes device-appropriate env vars to Iris."""

    def test_tpu_actor_gets_tpu_env_vars(self):
        """TPU actors must receive JAX_PLATFORMS='' and LIBTPU_INIT_ARGS from device defaults."""
        fake_iris = MagicMock()
        fake_iris.submit.return_value = MagicMock(job_id="job-tpu")
        client = FrayIrisClient.from_iris_client(fake_iris)

        class _DummyActor:
            pass

        resources = ResourceConfig.with_tpu("v5p-8")
        client.create_actor_group(_DummyActor, name="tpu-actor", count=1, resources=resources)

        kwargs = fake_iris.submit.call_args.kwargs
        env = kwargs["environment"]
        assert env is not None
        assert env.env_vars["JAX_PLATFORMS"] == ""
        assert "LIBTPU_INIT_ARGS" in env.env_vars

    def test_cpu_actor_gets_cpu_env_vars(self):
        """CPU actors must receive JAX_PLATFORMS=cpu."""
        fake_iris = MagicMock()
        fake_iris.submit.return_value = MagicMock(job_id="job-cpu")
        client = FrayIrisClient.from_iris_client(fake_iris)

        class _DummyActor:
            pass

        resources = ResourceConfig(cpu=2, ram="4g")
        client.create_actor_group(_DummyActor, name="cpu-actor", count=1, resources=resources)

        kwargs = fake_iris.submit.call_args.kwargs
        env = kwargs["environment"]
        assert env is not None
        assert env.env_vars["JAX_PLATFORMS"] == "cpu"

    def test_gpu_actor_gets_gpu_env_vars(self):
        """GPU actors must receive JAX_PLATFORMS='' from GpuConfig defaults."""
        fake_iris = MagicMock()
        fake_iris.submit.return_value = MagicMock(job_id="job-gpu")
        client = FrayIrisClient.from_iris_client(fake_iris)

        class _DummyActor:
            pass

        resources = ResourceConfig.with_gpu("a100-80g")
        client.create_actor_group(_DummyActor, name="gpu-actor", count=1, resources=resources)

        kwargs = fake_iris.submit.call_args.kwargs
        env = kwargs["environment"]
        assert env is not None
        assert env.env_vars["JAX_PLATFORMS"] == ""


class TestWithTpuFlexible:
    def test_single_type_returns_standard_config(self):
        rc = ResourceConfig.with_tpu(["v5p-8"])
        assert isinstance(rc.device, TpuConfig)
        assert rc.device.variant == "v5p-8"
        assert rc.device_alternatives is None

    def test_multiple_types_sets_alternatives(self):
        rc = ResourceConfig.with_tpu(["v4-8", "v5p-8"])
        assert rc.device.variant == "v4-8"
        assert rc.device_alternatives == ["v5p-8"]
        assert rc.replicas == 1  # both v4-8 and v5p-8 have vm_count=1

    def test_mismatched_vm_count_raises(self):
        with pytest.raises(ValueError, match="vm_count and chips_per_vm"):
            ResourceConfig.with_tpu(["v4-8", "v4-16"])

    def test_mismatched_chips_per_vm_raises(self):
        # v6e-4 and v6e-8 both have vm_count=1 but 4 vs 8 chips per VM;
        # the single VM of a v6e-8 is indivisible so these must not mix.
        with pytest.raises(ValueError, match="vm_count and chips_per_vm"):
            ResourceConfig.with_tpu(["v6e-4", "v6e-8"])

    def test_same_chips_per_vm_different_generations_ok(self):
        # v4-8 and v5p-8 both have vm_count=1 and chips_per_vm=4.
        rc = ResourceConfig.with_tpu(["v4-8", "v5p-8"])
        assert rc.device.variant == "v4-8"
        assert rc.device_alternatives == ["v5p-8"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ResourceConfig.with_tpu([])

    def test_slice_count_multiplies_replicas(self):
        rc = ResourceConfig.with_tpu(["v5p-16", "v4-16"], slice_count=2)
        # v5p-16 has vm_count=2, so replicas = 2 * 2 = 4
        assert rc.replicas == 4


# resolve_coscheduling: multi-host gangs pick the topology level the Iris provider maps.
# group_by is now a literal topology level (B4 rename); an unmapped value raises at K8s
# pod-manifest build, so the fray defaults must stay in sync with the provider's map.


def test_resolve_coscheduling_gpu_multinode_uses_leafgroup():
    cosched = resolve_coscheduling(GpuConfig(variant="H100", count=8), replicas=2)
    assert cosched is not None
    assert cosched.group_by == "leafgroup"


def test_resolve_coscheduling_tpu_multinode_uses_tpu_name():
    cosched = resolve_coscheduling(TpuConfig(variant="v5litepod-16"), replicas=4)
    assert cosched is not None
    assert cosched.group_by == "tpu-name"


def test_resolve_coscheduling_single_replica_is_none():
    assert resolve_coscheduling(GpuConfig(variant="H100", count=8), replicas=1) is None
    assert resolve_coscheduling(CpuConfig(), replicas=4) is None


def _gpu_resources(count: int) -> ResourceSpec:
    return ResourceSpec(cpu=4, memory="8GB", disk="16GB", device=gpu_device("H100", count))


def test_wrap_multiprocess_one_process_per_gpu() -> None:
    # fray composes the multigpu supervisor into the command; iris runs it verbatim.
    wrapped = wrap_multiprocess(
        IrisEntrypoint.from_command("python", "train.py", "--steps", "10"), _gpu_resources(8), processes_per_task=8
    )
    assert wrapped.command == [
        "python",
        "-m",
        "iris.hooks.multigpu_main",
        "--nproc",
        "8",
        "--devices-per-proc",
        "1",
        "--",
        "python",
        "train.py",
        "--steps",
        "10",
    ]


def test_wrap_multiprocess_groups_devices_when_fewer_processes() -> None:
    wrapped = wrap_multiprocess(
        IrisEntrypoint.from_command("python", "train.py"), _gpu_resources(8), processes_per_task=4
    )
    assert wrapped.command[:8] == [
        "python",
        "-m",
        "iris.hooks.multigpu_main",
        "--nproc",
        "4",
        "--devices-per-proc",
        "2",
        "--",
    ]


def test_wrap_multiprocess_requires_gpu() -> None:
    cpu_only = ResourceSpec(cpu=4, memory="8GB", disk="16GB", device=None)
    with pytest.raises(ValueError, match="requires a GPU device"):
        wrap_multiprocess(IrisEntrypoint.from_command("python", "x.py"), cpu_only, processes_per_task=2)


def test_wrap_multiprocess_requires_divisible_gpu_count() -> None:
    with pytest.raises(ValueError, match="must divide the GPU count"):
        wrap_multiprocess(IrisEntrypoint.from_command("python", "x.py"), _gpu_resources(8), processes_per_task=3)
