# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the fray v2 Iris backend.

Tests type conversions and handle serialization without requiring an Iris cluster.
Integration tests that need a running cluster are marked with @pytest.mark.iris.
"""

import pickle
import threading

import pytest

from fray.v2.iris_backend import (
    IrisActorHandle,
    _start_requested_actor_termination,
    convert_constraints,
)
from fray.v2.types import (
    ResourceConfig,
    TpuConfig,
)


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
        assert c.value == "false"

    def test_single_region_produces_eq_constraint(self):
        resources = ResourceConfig(regions=["us-central1"])
        constraints = convert_constraints(resources)
        region_constraints = [c for c in constraints if c.key == "region"]
        assert len(region_constraints) == 1
        c = region_constraints[0]
        from iris.cluster.constraints import ConstraintOp

        assert c.op == ConstraintOp.EQ
        assert c.value == "us-central1"

    def test_multiple_regions_produce_in_constraint(self):
        resources = ResourceConfig(regions=["us-central1", "us-central2"])
        constraints = convert_constraints(resources)
        region_constraints = [c for c in constraints if c.key == "region"]
        assert len(region_constraints) == 1
        c = region_constraints[0]
        from iris.cluster.constraints import ConstraintOp

        assert c.op == ConstraintOp.IN
        assert c.values == ("us-central1", "us-central2")


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
        from iris.cluster.constraints import ConstraintOp

        assert c.op == ConstraintOp.IN
        assert set(c.values) == {"v4-8", "v5p-8"}


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


def test_single_replica_termination_uses_job_termination(monkeypatch):
    calls: list[str] = []

    class ImmediateThread:
        def __init__(self, target, daemon: bool, name: str):
            self._target = target
            self.daemon = daemon
            self.name = name

        def start(self) -> None:
            self._target()

    monkeypatch.setattr(threading, "Thread", ImmediateThread)

    _start_requested_actor_termination(
        requested=True,
        replica_count=1,
        thread_name="terminate-actor-0",
        terminate_job=lambda: calls.append("job"),
        terminate_replica=lambda: calls.append("replica"),
    )

    assert calls == ["job"]


def test_multi_replica_termination_uses_replica_shutdown(monkeypatch):
    calls: list[str] = []

    class ImmediateThread:
        def __init__(self, target, daemon: bool, name: str):
            self._target = target
            self.daemon = daemon
            self.name = name

        def start(self) -> None:
            self._target()

    monkeypatch.setattr(threading, "Thread", ImmediateThread)

    _start_requested_actor_termination(
        requested=True,
        replica_count=2,
        thread_name="terminate-actor-0",
        terminate_job=lambda: calls.append("job"),
        terminate_replica=lambda: calls.append("replica"),
    )

    assert calls == ["replica"]


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
        with pytest.raises(ValueError, match="same vm_count"):
            ResourceConfig.with_tpu(["v4-8", "v4-16"])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ResourceConfig.with_tpu([])

    def test_slice_count_multiplies_replicas(self):
        rc = ResourceConfig.with_tpu(["v5p-16", "v4-16"], slice_count=2)
        # v5p-16 has vm_count=2, so replicas = 2 * 2 = 4
        assert rc.replicas == 4
