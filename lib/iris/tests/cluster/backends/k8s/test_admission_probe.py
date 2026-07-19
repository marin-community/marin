# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the ``iris.admission_probe`` emitter: probe outcomes at
the K8sService boundary, failure classification, and message truncation."""

import pytest
from iris.cluster.backends.k8s.admission_probe import (
    PROBE_MESSAGE_MAX_LEN,
    AdmissionProber,
    ProbeOutcome,
)
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import KubectlError
from iris.test_util import FakeStatsTable


@pytest.fixture
def prober_env():
    k8s = InMemoryK8sService(namespace="iris")
    table = FakeStatsTable()
    prober = AdmissionProber(k8s, table, poll_interval=3600)
    yield k8s, table, prober
    prober.close()


def latest_row(table: FakeStatsTable):
    (row,) = table.writes[-1]
    return row


def test_accepted_dry_run_emits_ok_row(prober_env):
    _, table, prober = prober_env
    prober.probe_once()
    row = latest_row(table)
    assert row.outcome == ProbeOutcome.OK
    assert row.namespace == "iris"
    assert row.error_class == ""
    assert row.message == ""


def test_fail_closed_webhook_emits_failed_webhook_row(prober_env):
    k8s, table, prober = prober_env
    k8s.inject_failure(
        "dry_run_create",
        KubectlError(
            "dry-run create Pod/iris-admission-probe failed (500): Internal Server Error "
            'Internal error occurred: failed calling webhook "mpod.kb.io": no endpoints available ' + "x" * 1000,
            status=500,
        ),
    )
    prober.probe_once()
    row = latest_row(table)
    assert row.outcome == ProbeOutcome.FAILED
    assert row.error_class == "webhook"
    assert 'failed calling webhook "mpod.kb.io"' in row.message
    assert len(row.message) <= PROBE_MESSAGE_MAX_LEN


@pytest.mark.parametrize(
    ("error", "expected_class"),
    [
        (KubectlError("dry-run create Pod/p failed (403): Forbidden", status=403), "forbidden"),
        (KubectlError("dry-run create Pod/p failed (429): Too Many Requests", status=429), "http_429"),
        (TimeoutError("Read timed out"), "timeout"),
        (ConnectionError("connection refused"), "unreachable"),
        (RuntimeError("something else entirely"), "error"),
    ],
)
def test_failure_classification(prober_env, error, expected_class):
    k8s, table, prober = prober_env
    k8s.inject_failure("dry_run_create", error)
    prober.probe_once()
    row = latest_row(table)
    assert row.outcome == ProbeOutcome.FAILED
    assert row.error_class == expected_class


def test_probe_recovers_after_failure(prober_env):
    """One-shot failures produce one failed row; the next probe reports ok again."""
    k8s, table, prober = prober_env
    k8s.inject_failure("dry_run_create", KubectlError("boom", status=500))
    prober.probe_once()
    prober.probe_once()
    assert [row.outcome for (row,) in table.writes] == [ProbeOutcome.FAILED, ProbeOutcome.OK]
