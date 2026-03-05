# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the fray v2 Iris backend.

Tests type conversions and handle serialization without requiring an Iris cluster.
Integration tests that need a running cluster are marked with @pytest.mark.iris.
"""

import pickle
from unittest.mock import MagicMock

from fray.v2.iris_backend import (
    FrayIrisClient,
    IrisActorHandle,
    convert_constraints,
)
from fray.v2.types import (
    Entrypoint,
    JobRequest,
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

    def test_single_region_produces_eq_constraint(self):
        resources = ResourceConfig(regions=["us-central1"])
        constraints = convert_constraints(resources)
        region_constraints = [c for c in constraints if c.key == "region"]
        assert len(region_constraints) == 1
        c = region_constraints[0]
        from iris.cluster.types import ConstraintOp

        assert c.op == ConstraintOp.EQ
        assert c.value == "us-central1"

    def test_multiple_regions_produce_in_constraint(self):
        resources = ResourceConfig(regions=["us-central1", "us-central2"])
        constraints = convert_constraints(resources)
        region_constraints = [c for c in constraints if c.key == "region"]
        assert len(region_constraints) == 1
        c = region_constraints[0]
        from iris.cluster.types import ConstraintOp

        assert c.op == ConstraintOp.IN
        assert c.values == ("us-central1", "us-central2")


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


class TestFrayIrisClientSubmit:
    def test_submit_forwards_retry_counts(self):
        iris = MagicMock()
        iris.submit.return_value = MagicMock(job_id="/u/job")

        client = FrayIrisClient.from_iris_client(iris)
        request = JobRequest(
            name="job",
            entrypoint=Entrypoint.from_binary("echo", ["hi"]),
            max_retries_failure=7,
            max_retries_preemption=13,
        )
        client.submit(request)

        kwargs = iris.submit.call_args.kwargs
        assert kwargs["max_retries_failure"] == 7
        assert kwargs["max_retries_preemption"] == 13

    def test_submit_forwards_default_retry_counts(self):
        iris = MagicMock()
        iris.submit.return_value = MagicMock(job_id="/u/job")

        client = FrayIrisClient.from_iris_client(iris)
        request = JobRequest(
            name="job-defaults",
            entrypoint=Entrypoint.from_binary("echo", ["hi"]),
        )
        client.submit(request)

        kwargs = iris.submit.call_args.kwargs
        assert kwargs["max_retries_failure"] == 0
        assert kwargs["max_retries_preemption"] == 100
