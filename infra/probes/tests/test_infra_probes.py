# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Federated scheduling checks through the public Marin controller."""

import json

from infra_probes import collect_federated_scheduling
from iris.cluster.constraints import Constraint, cluster_directive
from iris.rpc import controller_pb2, job_pb2
from runner import METRIC_UP


class FakePeerClient:
    def __init__(self, peer_ids: list[str]) -> None:
        self._response = controller_pb2.Controller.ListPeersResponse(
            peers=[controller_pb2.Controller.PeerSummary(peer_id=peer_id) for peer_id in peer_ids]
        )

    def list_peers(self, request):
        return self._response


class FakeIris:
    def __init__(self, failing_clusters: tuple[str, ...] = ()) -> None:
        self._failing_clusters = set(failing_clusters)
        self.submissions: dict[str, dict] = {}

    def submit_job(self, **kwargs):
        constraints = [Constraint.from_proto(constraint) for constraint in kwargs["constraints"]]
        cluster = cluster_directive(constraints)
        assert cluster is not None
        self.submissions[cluster] = kwargs
        return kwargs["job_id"]

    def wait_for_job(self, job_id, timeout):
        state = (
            job_pb2.JOB_STATE_FAILED
            if any(cluster in str(job_id) for cluster in self._failing_clusters)
            else job_pb2.JOB_STATE_SUCCEEDED
        )
        return job_pb2.JobStatus(state=state)


def _health_by_cluster(samples) -> dict[str, float]:
    health = {}
    for sample in samples:
        if sample.metric != METRIC_UP:
            continue
        labels = json.loads(sample.labels)
        health[labels["cluster"]] = sample.value
    return health


def _labels_by_cluster(samples) -> dict[str, dict[str, str]]:
    labels_by_cluster = {}
    for sample in samples:
        if sample.metric != METRIC_UP:
            continue
        labels = json.loads(sample.labels)
        labels_by_cluster[labels["cluster"]] = labels
    return labels_by_cluster


def test_federated_scheduling_targets_every_peer_with_tiny_no_setup_job():
    iris = FakeIris()
    samples = collect_federated_scheduling(
        iris,
        FakePeerClient(["cw-us-east-08a", "cw-rno2a", "cw-us-east-02a"]),
    )

    assert _health_by_cluster(samples) == {
        "cw-rno2a": 1.0,
        "cw-us-east-02a": 1.0,
        "cw-us-east-08a": 1.0,
    }
    assert set(iris.submissions) == set(_health_by_cluster(samples))
    assert _labels_by_cluster(samples) == {
        peer_id: {
            "cluster": peer_id,
            "probe": f"iris-job-submit/cluster/{peer_id}",
            "route": "federation",
        }
        for peer_id in iris.submissions
    }
    for submission in iris.submissions.values():
        assert submission["resources"].cpu_millicores == 100
        assert submission["resources"].memory_bytes == 128 * 1024**2
        assert list(submission["environment"].setup_scripts) == []


def test_federated_scheduling_preserves_each_peer_result_when_one_fails():
    iris = FakeIris(failing_clusters=("cw-rno2a",))

    samples = collect_federated_scheduling(
        iris,
        FakePeerClient(["cw-rno2a", "cw-us-east-02a"]),
    )

    assert _health_by_cluster(samples) == {
        "cw-rno2a": 0.0,
        "cw-us-east-02a": 1.0,
    }
