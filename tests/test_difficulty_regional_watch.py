# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise alert replay and exact-owned teardown without live cloud resources."""

import copy
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pyarrow as pa
import pytest
from iris.cluster.platforms.gcp.fake import InMemoryGcpService
from iris.cluster.platforms.gcp.service import VmCreateRequest
from iris.cluster.service_mode import ServiceMode
from iris.resources.state import JobState
from iris.rpc import controller_pb2
from rigging.filesystem.conditional_object import conditional_object

from scripts.ops import difficulty_regional_watch as watch

NOW = datetime.now(UTC)
RUNS = "gs://marin-us-central2/campaign/runs/"
SID = "marin-difficulty-cpu-test"


def event(attempt, *, outcome="unscored", reason="ENVIRONMENT_START_FAILURE", prefix=RUNS):
    row = {
        "attempt_id": attempt,
        "source_id": "source",
        "outcome": outcome,
        "reason": reason,
        "evidence_uri": prefix + attempt + "/attempt.json",
    }
    return {
        "cluster": "marin",
        "key": "/benfeuer/atlas-difficulty/task/0:0",
        "seq": len(attempt),
        "epoch_ms": int(NOW.timestamp() * 1000),
        "data": "INFO:DIFFICULTY_OUTCOME " + json.dumps(row),
    }


def test_real_producer_events_survive_restart_and_ignore_protocol_control_rows():
    state = {}
    end = int(NOW.timestamp() * 1000)
    records = watch.failure_tick(
        state,
        [
            event("one"),
            event("one"),
            event("prior", prefix="gs://other/runs/"),
            event("superseded", outcome="protocol_superseded"),
        ],
        RUNS,
        end,
        NOW,
    )
    assert len(records) == 1
    assert records[0]["evidence"]["count"] == 1
    assert records[0]["evidence"]["example_attempt_ids"] == ["one"]
    assert records[0]["evidence"]["finelog_examples"][0]["key"].endswith("/0:0")
    restarted = json.loads(json.dumps(state))
    assert (
        watch.failure_tick(restarted, [event("one"), event("two")], RUNS, end + 300000, NOW + timedelta(minutes=5)) == []
    )
    later = watch.failure_tick(restarted, [event("two"), event("three")], RUNS, end + 3600000, NOW + timedelta(hours=1))
    assert later[0]["evidence"]["count"] == 3
    assert later[0]["evidence"]["example_attempt_ids"] == ["one", "two", "three"]


def test_alert_outbox_replays_after_partial_upload_without_duplicate_objects(tmp_path, monkeypatch):
    state = {}
    records = watch.failure_tick(
        state, [event("one"), event("two", reason="OTHER")], RUNS, int(NOW.timestamp() * 1000), NOW
    )
    state["outbox"] = records
    persisted = json.dumps(state)
    real_write = watch.write_json
    stop = records[1]["event_id"]

    def interrupted_write(path, value):
        if stop in path:
            raise ConnectionError("regional storage interrupted")
        real_write(path, value)

    monkeypatch.setattr(watch, "write_json", interrupted_write)
    with pytest.raises(ConnectionError):
        watch.publish_outbox(state, str(tmp_path))
    restarted = json.loads(persisted)
    monkeypatch.setattr(watch, "write_json", real_write)
    watch.publish_outbox(restarted, str(tmp_path))
    outputs = [json.loads(path.read_text()) for path in (tmp_path / "events").glob("*.json")]
    assert {row["event_id"] for row in outputs} == {row["event_id"] for row in records}
    assert len(outputs) == 2
    assert restarted["outbox"] == []


def test_saturated_query_fails_without_advancing_the_durable_cursor():
    class Logs:
        def query(self, sql, *, max_rows):
            return pa.Table.from_pylist([event("x")] * (watch.EVENT_LIMIT + 1))

    state = {"cursor_ms": 1000000}
    with pytest.raises(RuntimeError, match="saturated"):
        events = watch.outcome_window(Logs(), 1, watch.WINDOW_MS)
        watch.failure_tick(state, events, RUNS, watch.WINDOW_MS, NOW)
    assert state == {"cursor_ms": 1000000}
    with pytest.raises(ValueError, match="bounded"):
        watch.outcome_window(Logs(), 0, watch.OVERLAP_MS + watch.WINDOW_MS + 1)


class Iris:
    def __init__(self, observations, state=JobState.RUNNING):
        self.observations = list(observations)
        self.state = state

    def job_status(self, job_id):
        return SimpleNamespace(state=self.state)

    def list_workers(self, query):
        if len(self.observations) > 1:
            return self.observations.pop(0)
        return self.observations[0]


def workers(*, jobs=(), healthy=True, age=0, sid=SID):
    return [
        controller_pb2.Controller.WorkerHealthStatus(
            worker_id=sid + "-worker-0",
            healthy=healthy,
            running_job_ids=jobs,
            last_heartbeat={"epoch_ms": int((NOW - timedelta(seconds=age)).timestamp() * 1000)},
        )
    ]


@pytest.fixture
def owned(tmp_path):
    ownership = {
        "owner": "/benfeuer/controller",
        "slices": [{"slice_id": SID, "zone": "us-central2-b", "state": "created"}],
    }
    fence = conditional_object(str(tmp_path / "ownership.json"))
    version = fence.write(json.dumps(ownership).encode(), expected_version=None)
    gcp = InMemoryGcpService(ServiceMode.DRY_RUN, project_id="test")
    gcp.vm_create(
        VmCreateRequest(
            name="exact-owned",
            zone="us-central2-b",
            machine_type="n2-highmem-4",
            labels={"iris-marin-slice-id": SID, "iris-marin-manual": "true"},
        )
    )
    gcp.vm_create(
        VmCreateRequest(
            name="peer",
            zone="us-central2-b",
            machine_type="n2-highmem-4",
            labels={"iris-marin-slice-id": "peer", "iris-marin-manual": "true"},
        )
    )
    return ownership, fence, version, gcp


def run_guard(owned, client, *, state=None, now=NOW, refresh=lambda: None):
    ownership, fence, version, gcp = owned
    if state is None:
        state = {}
    records = watch.guard_tick(state, ownership, client, gcp, fence, version, now, refresh)
    return state, records


@pytest.mark.parametrize("observation", [workers(jobs=("/peer/task/0",)), workers(healthy=False), workers(age=600), []])
def test_terminal_owner_does_not_delete_busy_unhealthy_stale_or_missing_workers(owned, observation):
    state, records = run_guard(owned, Iris([observation], JobState.SUCCEEDED))
    assert records == []
    assert owned[3].vm_describe("exact-owned", "us-central2-b") is not None
    assert not state["slices"][SID]["cleanup_due"]


def test_idle_hour_resets_after_work_then_deletes_only_exact_owned_vm(owned):
    state, _ = run_guard(owned, Iris([workers()]))
    state, _ = run_guard(
        owned, Iris([workers(jobs=("/benfeuer/grade/0",))]), state=state, now=NOW + timedelta(minutes=30)
    )
    assert state["slices"][SID]["idle_since"] is None
    # Fresh heartbeats are needed at each observation; model time at the boundary.
    later = NOW + timedelta(minutes=31)
    fresh = workers()
    fresh[0].last_heartbeat.epoch_ms = int(later.timestamp() * 1000)
    run_guard(owned, Iris([fresh]), state=state, now=later)
    assert owned[3].vm_describe("exact-owned", "us-central2-b") is not None
    end = later + timedelta(hours=1)
    first = copy.deepcopy(fresh)
    first[0].last_heartbeat.epoch_ms = int(end.timestamp() * 1000)
    # The guard rechecks using the actual wall clock immediately before delete.
    state, records = run_guard(owned, Iris([first, workers()]), state=state, now=end)
    assert records[0]["kind"] == "DIFFICULTY_CPU_SLICE_DELETE_REQUESTED"
    assert state["slices"][SID]["state"] == "delete_requested"
    assert owned[3].vm_describe("exact-owned", "us-central2-b") is None
    assert owned[3].vm_describe("peer", "us-central2-b") is not None
    state, records = run_guard(owned, Iris([[]]), state=state, now=end)
    assert state["slices"][SID]["state"] == "deleted"
    assert records[0]["kind"] == "DIFFICULTY_CPU_SLICE_DELETED"


def test_new_work_before_delete_preserves_vm(owned):
    state, records = run_guard(owned, Iris([workers(), workers(jobs=("/peer/work/0",))], JobState.SUCCEEDED))
    assert records == []
    assert owned[3].vm_describe("exact-owned", "us-central2-b") is not None
    assert state["slices"][SID]["idle_since"] is None


def test_transferred_ownership_prevents_deletion(owned):
    owned[1].write(b'{"owner":"new-controller"}', expected_version=owned[2])
    with pytest.raises(RuntimeError, match="fence changed"):
        run_guard(owned, Iris([workers()], JobState.SUCCEEDED))
    assert owned[3].vm_describe("exact-owned", "us-central2-b") is not None


def test_missing_delete_capability_still_records_every_owned_worker(owned):
    owned[0]["capability_error"] = "compute.instances.delete missing"
    state = {}
    with pytest.raises(PermissionError, match=r"compute\.instances\.delete"):
        run_guard(owned, Iris([workers()], JobState.SUCCEEDED), state=state)
    assert state["slices"][SID]["healthy"]
    assert state["slices"][SID]["cleanup_due"]
    assert state["slices"][SID]["state"] == "created"
    assert owned[3].vm_describe("exact-owned", "us-central2-b") is not None


def test_fresh_heartbeat_after_tick_started_is_not_misclassified_as_unhealthy():
    # The RPC can return a heartbeat newer than the timestamp captured before
    # a slow Finelog query. The guard must compare freshness at RPC receipt.
    native = workers()
    current = datetime.now(UTC)
    native[0].last_heartbeat.epoch_ms = int(current.timestamp() * 1000)
    observation = watch.slice_observation(Iris([native]), SID, current - timedelta(minutes=5))
    assert observation.healthy
