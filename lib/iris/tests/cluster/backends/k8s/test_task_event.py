# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``iris.task_event`` scheduling/admission timeline: the "event log for
every job". The k8s backend classifies a not-yet-running pod into a
``(source, reason)`` verdict and appends a finelog row when that verdict
changes, so the dashboard can render *why* a task is wedged in BUILDING
(Kueue admission denial, image-pull failure) as a sequence, not a single
opaque "Building 18m"."""

from iris.cluster.backends.k8s.tasks import (
    _EVENT_SOURCE_CONTAINER,
    _EVENT_SOURCE_KUEUE,
    _LABEL_ATTEMPT_ID,
    _LABEL_MANAGED,
    _LABEL_RUNTIME,
    _LABEL_TASK_HASH,
    _RUNTIME_LABEL_VALUE,
    TaskEventLog,
    _pod_event,
    _pod_group_name,
    _pod_name,
    _task_hash,
)
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.types import JobName
from iris.test_util import FakeStatsTable

from .conftest import (
    KUEUE_UNADMITTED_MSG,
    gated_pod,
    imagepull_pod,
    make_batch,
    make_kueue_provider,
    unadmitted_workload,
    unevaluated_workload,
)

# --- _pod_event classification -------------------------------------------------


def test_pod_event_classifies_blocked_kueue_admission_as_warning():
    """A gated pod whose Workload Kueue has declined is a Warning from k8s/kueue."""
    ev = _pod_event(gated_pod(), unadmitted_workload())
    assert ev is not None
    assert ev.source == _EVENT_SOURCE_KUEUE
    assert ev.reason == "SchedulingGated"
    assert ev.severity == "Warning"
    assert "couldn't assign flavors" in ev.message


def test_pod_event_gated_but_not_yet_evaluated_is_normal():
    """A freshly gated pod Kueue has not ruled on yet is Normal, not an alarm —
    only a positively-declined admission (QuotaReserved=False) reads as Warning."""
    ev = _pod_event(gated_pod(), unevaluated_workload())
    assert ev is not None
    assert ev.source == _EVENT_SOURCE_KUEUE
    assert ev.severity == "Normal"


def test_pod_event_image_pull_is_container_warning():
    ev = _pod_event(imagepull_pod(), None)
    assert ev is not None
    assert ev.source == _EVENT_SOURCE_CONTAINER
    assert ev.reason == "ImagePullBackOff"
    assert ev.severity == "Warning"
    assert "ghcr.io/nope" in ev.message


def test_pod_event_none_for_healthy_running_pod():
    pod = {
        "metadata": {"name": "iris-job-0-0"},
        "status": {"phase": "Running", "containerStatuses": [{"name": "task", "state": {"running": {}}}]},
    }
    assert _pod_event(pod, None) is None


# --- TaskEventLog dedup / retain ----------------------------------------------


def test_event_log_writes_once_per_verdict_and_dedups_message_drift():
    """One row on a new verdict; an unchanged (source, reason) is a no-op even
    when the message's numerals drift (Total nodes: 32 -> 40)."""
    table = FakeStatsTable()
    log = TaskEventLog(table)
    key = ("/job/0", 0)

    log.observe(key, _pod_event(gated_pod(), unadmitted_workload()))
    drifted = unadmitted_workload(msg=KUEUE_UNADMITTED_MSG.replace("Total nodes: 32", "Total nodes: 40"))
    log.observe(key, _pod_event(gated_pod(), drifted))
    log.observe(key, None)  # pod momentarily quiet — no row, verdict retained

    rows = [r for w in table.writes for r in w]
    assert len(rows) == 1
    assert rows[0].reason == "SchedulingGated"
    assert rows[0].count == 1


def test_event_log_records_a_severity_upgrade_under_the_same_reason():
    """A gated pod first seen before Kueue evaluates its Workload is Normal; once
    Kueue declines the same Workload it flips to Warning while (source, reason)
    stays (k8s/kueue, SchedulingGated). That actionable Warning must still record —
    the dedup keys on severity too, so the admission denial is never suppressed."""
    table = FakeStatsTable()
    log = TaskEventLog(table)
    key = ("/job/0", 0)

    log.observe(key, _pod_event(gated_pod(), unevaluated_workload()))
    log.observe(key, _pod_event(gated_pod(), unadmitted_workload()))

    rows = [r for w in table.writes for r in w]
    assert [(r.reason, r.source, r.type) for r in rows] == [
        ("SchedulingGated", _EVENT_SOURCE_KUEUE, "Normal"),
        ("SchedulingGated", _EVENT_SOURCE_KUEUE, "Warning"),
    ]
    assert "couldn't assign flavors" in rows[1].message


def test_event_log_appends_a_row_when_the_verdict_changes():
    table = FakeStatsTable()
    log = TaskEventLog(table)
    key = ("/job/0", 0)

    log.observe(key, _pod_event(gated_pod(), unadmitted_workload()))
    log.observe(key, _pod_event(imagepull_pod(), None))

    rows = [r for w in table.writes for r in w]
    assert [(r.reason, r.source) for r in rows] == [
        ("SchedulingGated", _EVENT_SOURCE_KUEUE),
        ("ImagePullBackOff", _EVENT_SOURCE_CONTAINER),
    ]


def test_event_log_retain_forgets_gone_attempts():
    """After an attempt leaves the running set, the same verdict re-emits — so a
    retried attempt starts its timeline clean rather than being deduped away."""
    table = FakeStatsTable()
    log = TaskEventLog(table)
    key = ("/job/0", 0)

    log.observe(key, _pod_event(gated_pod(), unadmitted_workload()))
    log.retain(set())  # attempt gone
    log.observe(key, _pod_event(gated_pod(), unadmitted_workload()))

    rows = [r for w in table.writes for r in w]
    assert len(rows) == 2


# --- end-to-end through sync() -------------------------------------------------


def _seed_gated_task(k8s, task_id: JobName, attempt_id: int = 0) -> None:
    """Seed a SchedulingGated, Kueue-unadmitted pod + its Workload, discoverable by sync()."""
    pod_group = _pod_group_name(task_id, attempt_id)
    pod = gated_pod(name=_pod_name(task_id, attempt_id), pod_group=pod_group)
    pod["kind"] = "Pod"
    pod["metadata"]["labels"].update(
        {
            _LABEL_MANAGED: "true",
            _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
            _LABEL_TASK_HASH: _task_hash(task_id.to_wire()),
            _LABEL_ATTEMPT_ID: str(attempt_id),
        }
    )
    k8s.seed_resource(K8sResource.PODS, pod["metadata"]["name"], pod)
    k8s.seed_resource(K8sResource.WORKLOADS, pod_group, unadmitted_workload(pod_group))


def test_sync_writes_the_kueue_verdict_to_the_event_log(k8s):
    """The full producer path: a running task whose pod is stuck SchedulingGated
    lands one Warning row carrying the Kueue admission verdict in iris.task_event."""
    event_table = FakeStatsTable()
    provider = make_kueue_provider(k8s, task_event_table=event_table)
    try:
        task_id = JobName.from_wire("/job/0")
        _seed_gated_task(k8s, task_id)
        entry = RunningTaskEntry(task_id=task_id, attempt_id=0)

        provider.sync(make_batch(running_tasks=[entry]))

        rows = [r for w in event_table.writes for r in w]
        assert len(rows) == 1
        row = rows[0]
        assert row.task_id == "/job/0"
        assert row.attempt_id == 0
        assert row.type == "Warning"
        assert row.reason == "SchedulingGated"
        assert row.source == _EVENT_SOURCE_KUEUE
        assert "couldn't assign flavors" in row.message

        # A second identical tick must not append a duplicate (verdict unchanged).
        provider.sync(make_batch(running_tasks=[entry]))
        rows = [r for w in event_table.writes for r in w]
        assert len(rows) == 1
    finally:
        provider.close()
