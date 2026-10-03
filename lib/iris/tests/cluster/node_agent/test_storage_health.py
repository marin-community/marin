# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import threading
from unittest.mock import patch

import pytest
from botocore.exceptions import EndpointConnectionError, NoCredentialsError, ReadTimeoutError
from fsspec.implementations.memory import MemoryFileSystem
from iris.cluster.config import NodeStorageHealthConfig
from iris.cluster.node_agent.storage_health import (
    CORDON_ANNOTATION,
    HEALTH_ANNOTATION,
    ProbeResult,
    StorageHealthReport,
    bounded_probe,
    probe_storage,
    reconcile_storage_health,
    run_storage_health,
    target_id,
)
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from rigging.timing import Timestamp


@pytest.fixture
def config():
    return NodeStorageHealthConfig(scratch_prefix="s3://regional/health", failure_threshold=3, interval=1, timeout=1)


def seed_node(k8s, config, name, result=ProbeResult.HEALTHY, *, failures=0, age=0, cordoned=False):
    now = Timestamp.now().epoch_seconds()
    report = StorageHealthReport(
        node_uid=name,
        boot_id="boot",
        target=target_id(config),
        started_at=now - age,
        checked_at=now - age,
        failure_since=now - 10 if failures else 0,
        failures=failures,
        result=result,
    )
    k8s.seed_resource(
        K8sResource.NODES,
        name,
        {
            "metadata": {
                "name": name,
                "uid": name,
                "resourceVersion": "1",
                "annotations": {HEALTH_ANNOTATION: report.model_dump_json(), "operator": "keep"},
            },
            "status": {"nodeInfo": {"bootID": "boot"}},
            "spec": {"unschedulable": cordoned, "taints": [{"key": "operator"}]},
        },
    )


def test_isolated_failure_cordons_only_after_threshold_and_preserves_node(config):
    k8s = InMemoryK8sService()
    seed_node(k8s, config, "bad", ProbeResult.FAILED, failures=2)
    seed_node(k8s, config, "healthy1")
    seed_node(k8s, config, "healthy2")
    reconcile_storage_health(k8s, config)
    assert not k8s.get_json(K8sResource.NODES, "bad")["spec"]["unschedulable"]
    seed_node(k8s, config, "bad", ProbeResult.FAILED, failures=3)
    reconcile_storage_health(k8s, config)
    node = k8s.get_json(K8sResource.NODES, "bad")
    assert node["spec"] == {"unschedulable": True, "taints": [{"key": "operator"}]}
    assert node["metadata"]["annotations"]["operator"] == "keep"
    assert CORDON_ANNOTATION in node["metadata"]["annotations"]
    # Healthy probes never erase either an operator or automatic cordon.
    seed_node(k8s, config, "manual", cordoned=True)
    recovered = StorageHealthReport.model_validate_json(node["metadata"]["annotations"][HEALTH_ANNOTATION])
    recovered.result = ProbeResult.HEALTHY
    recovered.failures = 0
    recovered.failure_since = 0
    k8s.patch_node("bad", {"metadata": {"annotations": {HEALTH_ANNOTATION: recovered.model_dump_json()}}})
    reconcile_storage_health(k8s, config)
    assert k8s.get_json(K8sResource.NODES, "bad")["spec"]["unschedulable"]
    assert k8s.get_json(K8sResource.NODES, "manual")["spec"]["unschedulable"]


@pytest.mark.parametrize("peer_fault", ["outage", "stale", "reboot", "wrong_target", "before_failure", "missing"])
def test_unreliable_peer_evidence_does_not_cordon(config, peer_fault):
    k8s = InMemoryK8sService()
    seed_node(k8s, config, "bad", ProbeResult.FAILED, failures=3)
    for name in ("p1", "p2"):
        seed_node(
            k8s,
            config,
            name,
            ProbeResult.FAILED if peer_fault == "outage" else ProbeResult.HEALTHY,
            age=100 if peer_fault == "stale" else 0,
        )
        node = k8s.get_json(K8sResource.NODES, name)
        if peer_fault == "reboot":
            node["status"]["nodeInfo"]["bootID"] = "new-boot"
        if peer_fault == "missing":
            node["metadata"]["annotations"].pop(HEALTH_ANNOTATION)
        if peer_fault in ("wrong_target", "before_failure"):
            report = StorageHealthReport.model_validate_json(node["metadata"]["annotations"][HEALTH_ANNOTATION])
            if peer_fault == "wrong_target":
                report.target = "different"
            else:
                report.started_at -= 30
            node["metadata"]["annotations"][HEALTH_ANNOTATION] = report.model_dump_json()
        k8s.seed_resource(K8sResource.NODES, name, node)
    reconcile_storage_health(k8s, config)
    assert not k8s.get_json(K8sResource.NODES, "bad")["spec"]["unschedulable"]


def test_cordon_budget_survives_reconciliation_and_manual_uncordon(config):
    k8s = InMemoryK8sService()
    for name in ("bad1", "bad2"):
        seed_node(k8s, config, name, ProbeResult.FAILED, failures=3)
    for name in ("p1", "p2", "p3"):
        seed_node(k8s, config, name)
    reconcile_storage_health(k8s, config)
    cordoned = [n for n in k8s.list_json(K8sResource.NODES) if n["spec"]["unschedulable"]]
    assert len(cordoned) == 1
    k8s.patch_node(cordoned[0]["metadata"]["name"], {"spec": {"unschedulable": False}})
    reconcile_storage_health(k8s, config)
    assert not any(n["spec"]["unschedulable"] for n in k8s.list_json(K8sResource.NODES))


def test_storage_probe_cleans_object_after_success_and_failed_read():
    fs = MemoryFileSystem()
    for fail in (False, True):
        with patch("iris.cluster.node_agent.storage_health.url_to_fs", return_value=(fs, "/health-probe")):
            if fail:
                with patch.object(fs, "cat_file", side_effect=OSError("unreachable")):
                    assert probe_storage("memory:///health-probe") == ProbeResult.FAILED
            else:
                assert probe_storage("memory:///health-probe") == ProbeResult.HEALTHY
        assert not fs.exists("/health-probe")


def test_bounded_probe_timeout_and_broken_process_are_distinct():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("probe", 1)):
        assert bounded_probe("s3://regional/health/test", 1) == ProbeResult.FAILED
    with patch("subprocess.run", return_value=subprocess.CompletedProcess("probe", 1)):
        assert bounded_probe("s3://regional/health/test", 1) == ProbeResult.CONFIGURATION_ERROR


def test_agent_resets_failure_streak_after_success_or_configuration_error(config):
    k8s = InMemoryK8sService()
    seed_node(k8s, config, "node")
    stop = threading.Event()
    results = iter(
        [
            ProbeResult.FAILED,
            ProbeResult.FAILED,
            ProbeResult.HEALTHY,
            ProbeResult.FAILED,
            ProbeResult.CONFIGURATION_ERROR,
            ProbeResult.FAILED,
        ]
    )
    observed = []

    def completed_wait(_interval):
        report = StorageHealthReport.model_validate_json(
            k8s.get_json(K8sResource.NODES, "node")["metadata"]["annotations"][HEALTH_ANNOTATION]
        )
        observed.append(report.failures)
        if len(observed) == 6:
            stop.set()

    with (
        patch("subprocess.run", side_effect=lambda *args, **kwargs: subprocess.CompletedProcess("probe", next(results))),
        patch.object(stop, "wait", side_effect=completed_wait),
    ):
        run_storage_health(k8s, "node", config, stop)
    assert observed == [1, 2, 0, 1, 0, 1]


@pytest.mark.parametrize(
    "error,expected",
    [
        (EndpointConnectionError(endpoint_url="https://regional.example"), ProbeResult.FAILED),
        (ReadTimeoutError(endpoint_url="https://regional.example"), ProbeResult.FAILED),
        (NoCredentialsError(), ProbeResult.CONFIGURATION_ERROR),
    ],
)
def test_storage_probe_network_failures_count_but_missing_credentials_do_not(error, expected):
    fs = MemoryFileSystem()
    with patch("iris.cluster.node_agent.storage_health.url_to_fs", return_value=(fs, "/classification-probe")):
        with patch.object(fs, "cat_file", side_effect=error):
            assert probe_storage("memory:///classification-probe") == expected
    assert not fs.exists("/classification-probe")
