# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run incremental failure alerts and an exact-owned CPU guard inside Iris.

The regional ownership object is the guard's allowlist and handover fence. Update
it before transferring slices to another controller; a changed version disables
all deletion until a new monitor is submitted with that version. This job never
allocates capacity, changes IAM, or scans ledger partitions.
"""

import argparse
import copy
import hashlib
import json
import logging
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import httpx
from finelog.client.log_client import LogClient
from iris.client.client import IrisClient, iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.platforms.gcp.cloud import CloudGcpService
from iris.cluster.platforms.gcp.service import GcpService
from iris.cluster.platforms.types import Labels
from iris.cluster.types import JobName
from iris.resources.state import TERMINAL_JOB_STATES
from iris.rpc import controller_pb2
from rigging.filesystem.conditional_object import ConditionalObject, conditional_object
from rigging.filesystem.distributed_lock import create_lock, lease_refresh
from rigging.filesystem.storage_path import StoragePath

from scripts.ops.difficulty_failure_watch import FailurePattern, campaign_outcome, update_patterns

LOGGER = logging.getLogger(__name__)
INTERVAL_SECONDS = 300
OVERLAP_MS = 3600_000
WINDOW_MS = 600_000
EVENT_LIMIT = 20000
STATE_LIMIT_BYTES = 32 * 1024 * 1024
MAX_PATTERNS = 10000
MAX_SEEN = 100000
IDLE_SECONDS = 3600
HEARTBEAT_MAX_AGE = 90


@dataclass(frozen=True)
class SliceObservation:
    healthy: bool
    jobs: tuple[str, ...]
    workers: tuple[str, ...]


def outcome_window(logs: LogClient, start_ms: int, end_ms: int) -> list[dict]:
    """Read at most one hour of overlap plus one ten-minute increment."""
    if not 0 < end_ms - start_ms <= OVERLAP_MS + WINDOW_MS:
        raise ValueError("Invalid bounded Finelog window")
    sql = (
        'SELECT key,seq,epoch_ms,data,"cluster" FROM "log" '
        "WHERE key LIKE '/benfeuer/atlas-difficulty%' "
        f"AND epoch_ms >= {start_ms} AND epoch_ms < {end_ms} "
        "AND contains(data,'DIFFICULTY_OUTCOME ') "
        f'ORDER BY epoch_ms,"cluster",key,seq LIMIT {EVENT_LIMIT + 1}'
    )
    rows = logs.query(sql, max_rows=EVENT_LIMIT + 1).to_pylist()
    if len(rows) > EVENT_LIMIT:
        raise RuntimeError("Finelog window saturated; cursor retained, no events silently dropped")
    return rows


def failure_tick(state: dict, events: list[dict], run_prefix: str, end_ms: int, now: datetime) -> list[dict]:
    """Apply a complete event window; retain replay-safe counts and transport IDs."""
    patterns = {key: FailurePattern(**value) for key, value in state.get("patterns", {}).items()}
    seen_at = state.setdefault("seen_at", {})
    seen = set(seen_at)
    outcomes = []
    references = state.setdefault("examples", {})
    for event in events:
        row = json.loads(event["data"].split("DIFFICULTY_OUTCOME ", 1)[1])
        if not campaign_outcome(row, run_prefix):
            continue
        outcomes.append(row)
        if row["attempt_id"] not in seen:
            references.setdefault(row["attempt_id"], {key: event[key] for key in ("cluster", "key", "seq", "epoch_ms")})
    alerts = update_patterns(patterns, seen, outcomes, now)
    for attempt in seen:
        seen_at.setdefault(attempt, end_ms)
    state["seen_at"] = {attempt: at for attempt, at in seen_at.items() if at >= end_ms - OVERLAP_MS - WINDOW_MS}
    state["patterns"] = {key: asdict(value) for key, value in patterns.items()}
    examples = {attempt for pattern in patterns.values() for attempt in pattern.examples}
    state["examples"] = {attempt: ref for attempt, ref in references.items() if attempt in examples}
    if len(patterns) > MAX_PATTERNS or len(state["seen_at"]) > MAX_SEEN:
        raise RuntimeError("Failure monitor state cap reached; refusing incomplete accounting")
    state["cursor_ms"] = end_ms
    records = []
    for alert in alerts:
        record = asdict(alert)
        record["at"] = now.isoformat()
        record["evidence"]["finelog_examples"] = [
            state["examples"][attempt] for attempt in alert.evidence["example_attempt_ids"]
        ]
        record["event_id"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        records.append(record)
    return records


def slice_observation(client: IrisClient, slice_id: str, now: datetime) -> SliceObservation:
    """Fail closed on missing, stale, unhealthy or non-exact worker identities."""
    workers = client.list_workers(controller_pb2.Controller.WorkerQuery(contains=slice_id, limit=10))
    # A heartbeat can advance while Finelog/RPC work runs. Compare at receipt,
    # not with the tick's earlier timestamp, or fresh workers look future-dated.
    now = max(now, datetime.now(UTC))
    # A manual CPU slice has exactly one worker. A partial page is never empty proof.
    exact = [worker for worker in workers if worker.worker_id == slice_id + "-worker-0"]
    healthy = len(workers) == len(exact) == 1 and all(
        worker.healthy and 0 <= now.timestamp() - worker.last_heartbeat.epoch_ms / 1000 <= HEARTBEAT_MAX_AGE
        for worker in exact
    )
    return SliceObservation(
        healthy=healthy,
        jobs=tuple(sorted({job for worker in workers for job in worker.running_job_ids})),
        workers=tuple(worker.worker_id for worker in workers),
    )


def cleanup_due(item: dict, observation: SliceObservation, root_terminal: bool, now: datetime) -> bool:
    """Require a terminal owner or a continuous healthy empty hour."""
    item.update(
        checked_at=now.isoformat(),
        healthy=observation.healthy,
        running_jobs=list(observation.jobs),
        worker_ids=list(observation.workers),
        root_terminal=root_terminal,
    )
    if not observation.healthy or observation.jobs:
        item["idle_since"] = None
        item["cleanup_due"] = False
        return False
    item.setdefault("idle_since", None)
    if item["idle_since"] is None:
        item["idle_since"] = now.isoformat()
    elapsed = (now - datetime.fromisoformat(item["idle_since"])).total_seconds()
    item["cleanup_due"] = root_terminal or elapsed >= IDLE_SECONDS
    return item["cleanup_due"]


def guard_tick(
    state: dict,
    ownership: dict,
    client: IrisClient,
    gcp: GcpService,
    fence: ConditionalObject,
    fence_version: str,
    now: datetime,
    refresh_lease,
) -> list[dict]:
    """Observe only allowlisted manual CPU VMs; recheck before maintained deletion."""
    root = client.job_status(JobName.from_string(ownership["owner"]))
    root_terminal = root.state in TERMINAL_JOB_STATES
    records = []
    due = []
    labels = Labels("marin")
    for seed in ownership["slices"]:
        sid = seed["slice_id"]
        item = state.setdefault("slices", {}).setdefault(sid, copy.deepcopy(seed))
        if item["state"] == "deleted":
            continue
        if item["state"] == "delete_requested":
            matches = gcp.vm_list(["us-central2-b"], {labels.iris_slice_id: sid, labels.iris_manual: "true"})
            if not matches:
                item.update(state="deleted", deleted_at=now.isoformat())
                records.append(
                    {
                        "kind": "DIFFICULTY_CPU_SLICE_DELETED",
                        "slice_id": sid,
                        "owner": ownership["owner"],
                        "at": now.isoformat(),
                    }
                )
            continue
        observation = slice_observation(client, sid, now)
        if cleanup_due(item, observation, root_terminal, now):
            due.append((sid, item))
    if ownership.get("capability_error"):
        raise PermissionError(ownership["capability_error"])
    for sid, item in due:
        # Exact label filter and zone; never list all project/TPU slices.
        matches = gcp.vm_list(["us-central2-b"], {labels.iris_slice_id: sid, labels.iris_manual: "true"})
        if len(matches) != 1:
            raise RuntimeError(f"Owned CPU slice has no unique manual VM: {sid}")
        vm = matches[0]
        if vm.labels.get(labels.iris_slice_id) != sid or vm.labels.get(labels.iris_manual) != "true":
            raise RuntimeError(f"Owned CPU VM labels changed: {sid}")
        fresh = slice_observation(client, sid, datetime.now(UTC))
        if not fresh.healthy or fresh.jobs:
            item["idle_since"] = None
            item["cleanup_due"] = False
            continue
        refresh_lease()
        if fence.version() != fence_version:
            raise RuntimeError("CPU ownership fence changed; refusing deletion")
        # CloudGcpService owns authenticated pooled HTTP and idempotent 404 handling.
        # A worker SA without delete permission fails here; never mark that VM deleted.
        gcp.vm_delete(vm.name, "us-central2-b", wait=False)
        # Record an accepted request, not a confirmed deletion. A later tick must
        # confirm absence from the exact filtered cloud inventory.
        item.update(
            state="delete_requested",
            delete_requested_at=datetime.now(UTC).isoformat(),
            vm_name=vm.name,
            cleanup_reason="owner terminal and empty" if root_terminal else "healthy empty for >=1hour",
        )
        records.append(
            {
                "kind": "DIFFICULTY_CPU_SLICE_DELETE_REQUESTED",
                "slice_id": sid,
                "vm_name": vm.name,
                "owner": ownership["owner"],
                "at": item["delete_requested_at"],
            }
        )
    return records


def write_json(path: str, value: dict) -> None:
    StoragePath(path).write_bytes(encoded_state(value))


def encoded_state(value: dict) -> bytes:
    payload = json.dumps(value, sort_keys=True).encode()
    if len(payload) > STATE_LIMIT_BYTES:
        raise RuntimeError("Regional state exceeds bounded object size")
    return payload


def publish_outbox(state: dict, root: str) -> None:
    """Replay durable event IDs after a crash without creating duplicate objects."""
    for record in state["outbox"]:
        write_json(root + "/events/" + record["event_id"] + ".json", record)
        LOGGER.warning("%s %s", record["kind"], json.dumps(record, sort_keys=True))
    state["outbox"] = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--ownership", required=True)
    parser.add_argument("--ownership-sha256", required=True)
    parser.add_argument("--duration-seconds", type=int, default=86400)
    parser.add_argument("--interval-seconds", type=int, default=INTERVAL_SECONDS)
    parser.add_argument("--once", action="store_true", help="Run one bounded regional deployment preflight")
    args = parser.parse_args()
    ctx = iris_ctx()
    info = get_job_info()
    assert info is not None and info.credentials is not None
    if info.worker_region != "us-central2" or not args.root.startswith("gs://marin-us-central2/"):
        raise ValueError("This monitor must execute and persist in us-central2")
    if args.interval_seconds < INTERVAL_SECONDS or not 300 <= args.duration_seconds <= 86400:
        raise ValueError("Require >=5-minute interval and a bounded <=24-hour job")
    fence = conditional_object(args.ownership)
    owned = fence.read()
    if owned is None or hashlib.sha256(owned.data).hexdigest() != args.ownership_sha256:
        raise ValueError("Ownership seed is absent or does not match the submitted allowlist")
    ownership = json.loads(owned.data)
    seeds = ownership["slices"]
    if len(seeds) > 32 or len({item["slice_id"] for item in seeds}) != len(seeds):
        raise ValueError("Invalid bounded owned CPU allowlist")
    if any(
        not item["slice_id"].startswith("marin-difficulty-cpu-") or item["zone"] != "us-central2-b" for item in seeds
    ):
        raise ValueError("Ownership seed includes another lane or region")
    deadline = time.monotonic() + args.duration_seconds
    state_object = conditional_object(args.root + "/state.json")
    with closing(
        LogClient.connect(
            LOG_SERVER_ENDPOINT_NAME,
            resolver=ctx.client.resolve_endpoint,
            interceptors=info.credentials.interceptors(),
            timeout_ms=35000,
        )
    ) as logs:
        # Discover schema once, then reuse this single RPC client's pooled transport.
        LOGGER.info("DIFFICULTY_MONITOR_LOG_SCHEMA %s", logs.get_table_schema("log"))
        with httpx.Client(timeout=30) as http:
            gcp = CloudGcpService("hai-gcp-models", http_client=http)
            while time.monotonic() < deadline:
                if fence.version() != owned.version:
                    raise RuntimeError("CPU ownership was transferred; stop and rebind monitor")
                lease = create_lock(args.root + "/monitor.lock")
                if not lease.try_acquire():
                    raise RuntimeError("Another regional monitor owns this state")
                try:
                    with lease_refresh(lease):
                        previous = state_object.read()
                        state = (
                            json.loads(previous.data)
                            if previous
                            else {"owner": ownership["owner"], "run_prefix": args.run_prefix}
                        )
                        if state["owner"] != ownership["owner"] or state["run_prefix"] != args.run_prefix:
                            raise ValueError("Monitor state belongs to another campaign/controller")
                        now = datetime.now(UTC)
                        upper = int(now.timestamp() * 1000) - 60_000
                        cursor = state.get("cursor_ms", upper - WINDOW_MS)
                        end = min(upper, cursor + WINDOW_MS)
                        events = outcome_window(logs, max(0, cursor - OVERLAP_MS), end)
                        records = failure_tick(state, events, args.run_prefix, end, now)
                        # Persist alert outbox WITH cursor/counts before emission. Retries replay
                        # the same IDs; immutable per-event keys make delivery idempotent.
                        state.setdefault("outbox", []).extend(records)
                        version = state_object.write(
                            encoded_state(state),
                            expected_version=previous.version if previous else None,
                        )
                        publish_outbox(state, args.root)
                        # Guard failures must stay visible without interrupting failure alerts.
                        try:
                            deleted = guard_tick(
                                state, ownership, ctx.client, gcp, fence, owned.version, now, lease.refresh
                            )
                            state["guard_error"] = ownership.get("capability_error")
                            for record in deleted:
                                write_json(args.root + "/deletions/" + record["slice_id"] + ".json", record)
                        except Exception as error:
                            state["guard_error"] = f"{type(error).__name__}: {error}"
                            LOGGER.exception("DIFFICULTY_CPU_GUARD_BLOCKED")
                        state.update(
                            checked_at=now.isoformat(),
                            input_events=len(events),
                            cursor_lag_seconds=(upper - end) / 1000,
                            forwarded_overlap_seconds=OVERLAP_MS / 1000,
                        )
                        # Lease serializes writers; the CAS still catches an external edit.
                        state_object.write(encoded_state(state), expected_version=version)
                        write_json(
                            args.root + "/heartbeat.json",
                            {
                                key: state[key]
                                for key in ("owner", "checked_at", "input_events", "cursor_lag_seconds", "guard_error")
                            },
                        )
                        LOGGER.info(
                            "DIFFICULTY_REGIONAL_MONITOR_TICK %s",
                            json.dumps(
                                {
                                    "events": len(events),
                                    "alerts": len(records),
                                    "guard_error": state["guard_error"],
                                    "slices": state.get("slices", {}),
                                },
                                sort_keys=True,
                            ),
                        )
                finally:
                    lease.release()
                if args.once:
                    break
                time.sleep(args.interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
