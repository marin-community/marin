# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the fray v2 Iris backend.

Tests type conversions and handle serialization without requiring an Iris cluster.
Integration tests that need a running cluster are marked with @pytest.mark.iris.
"""

import pickle

from fray.v2.iris_backend import (
    IrisActorHandle,
    convert_constraints,
)
from fray.v2.types import (
    ResourceConfig,
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
