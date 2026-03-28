# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-tick dispatch logic: query status, decide, launch agent, persist."""

import fcntl
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from marin.dispatch.agent_adapter import AgentResult, AgentSession
from marin.dispatch.git_ops import append_logbook, cleanup_worktree, commit_and_push, setup_worktree
from marin.dispatch.github_ops import post_escalation, post_progress_comment
from marin.dispatch.schema import (
    RayRunConfig,
    RunPointer,
    RunState,
    RunStatus,
    TickEvent,
    TickEventKind,
)
from marin.dispatch.storage import load_collection, load_state, save_state

logger = logging.getLogger(__name__)

ESCALATION_THRESHOLD = 3

RAY_STATUS_MAP = {
    "PENDING": RunStatus.PENDING,
    "RUNNING": RunStatus.RUNNING,
    "SUCCEEDED": RunStatus.SUCCEEDED,
    "FAILED": RunStatus.FAILED,
    "STOPPED": RunStatus.STOPPED,
}

IRIS_STATUS_MAP = {
    "JOB_STATE_PENDING": RunStatus.PENDING,
    "JOB_STATE_BUILDING": RunStatus.PENDING,
    "JOB_STATE_RUNNING": RunStatus.RUNNING,
    "JOB_STATE_SUCCEEDED": RunStatus.SUCCEEDED,
    "JOB_STATE_FAILED": RunStatus.FAILED,
    "JOB_STATE_KILLED": RunStatus.STOPPED,
    "JOB_STATE_WORKER_FAILED": RunStatus.FAILED,
    "JOB_STATE_UNSCHEDULABLE": RunStatus.FAILED,
}


@dataclass
class TickOutcome:
    collection_name: str
    dispatched: int = 0
    succeeded: int = 0
    failed: int = 0
    escalated: int = 0


def query_ray_status(run: RayRunConfig) -> RunStatus:
    try:
        from marin.run.ray_run import make_client

        client = make_client()
        status = str(client.get_job_status(run.job_id))
        return RAY_STATUS_MAP.get(status.upper(), RunStatus.UNKNOWN)
    except Exception:
        logger.warning("Failed to query Ray job %s", run.job_id, exc_info=True)
        return RunStatus.UNKNOWN


def _fetch_iris_jobs() -> dict[str, RunStatus]:
    """Fetch all Iris jobs once and return a mapping of job_id/name -> status."""
    try:
        result = subprocess.run(
            ["iris", "job", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("iris job list failed: %s", result.stderr.strip())
            return {}
        jobs = json.loads(result.stdout)
        status_by_id: dict[str, RunStatus] = {}
        for job in jobs:
            state = job.get("state", "").upper()
            status = IRIS_STATUS_MAP.get(state, RunStatus.UNKNOWN)
            for key in ("job_id", "name"):
                if key in job:
                    status_by_id[job[key]] = status
        return status_by_id
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        return {}


def query_run_status(
    run_pointer: RunPointer,
    iris_cache: dict[str, RunStatus] | None = None,
) -> RunStatus:
    if run_pointer.ray is not None:
        return query_ray_status(run_pointer.ray)
    if run_pointer.iris is not None:
        if iris_cache is not None:
            return iris_cache.get(run_pointer.iris.job_id, RunStatus.UNKNOWN)
        return _fetch_iris_jobs().get(run_pointer.iris.job_id, RunStatus.UNKNOWN)
    return RunStatus.UNKNOWN


def should_dispatch(
    current_status: RunStatus,
    previous_status: RunStatus,
    event_kind: TickEventKind,
    consecutive_failures: int,
) -> bool:
    if consecutive_failures >= ESCALATION_THRESHOLD:
        return False
    if event_kind == TickEventKind.MANUAL:
        return True
    if event_kind == TickEventKind.FAILURE_ALERT:
        return current_status in (RunStatus.FAILED, RunStatus.STOPPED)
    # Scheduled poll: dispatch on status change or for running health checks.
    if current_status != previous_status:
        return True
    if current_status == RunStatus.RUNNING:
        return True
    return False


def _lock_path(repo_root: Path, collection_name: str) -> Path:
    lock_dir = repo_root / ".agents" / "collections"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / f"{collection_name}.lock"


def process_tick(
    collection_name: str,
    event_kind: TickEventKind,
    agent: AgentSession,
    repo_root: Path,
) -> TickOutcome:
    outcome = TickOutcome(collection_name=collection_name)

    lock_file = _lock_path(repo_root, collection_name)
    lock_fd = open(lock_file, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.warning("Collection %s is already being processed, skipping", collection_name)
        lock_fd.close()
        return outcome

    try:
        outcome = _process_tick_locked(collection_name, event_kind, agent, repo_root)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    return outcome


def _process_tick_locked(
    collection_name: str,
    event_kind: TickEventKind,
    agent: AgentSession,
    repo_root: Path,
) -> TickOutcome:
    outcome = TickOutcome(collection_name=collection_name)
    collection = load_collection(repo_root, collection_name)

    if collection.paused:
        logger.info("Collection %s is paused, skipping", collection_name)
        return outcome

    states = load_state(repo_root, collection_name)
    while len(states) < len(collection.runs):
        states.append(RunState())

    now = datetime.now(timezone.utc).isoformat()
    worktree_path: Path | None = None

    # Fetch iris jobs once for all runs in this collection.
    has_iris = any(rp.iris is not None for rp in collection.runs)
    iris_cache = _fetch_iris_jobs() if has_iris else None

    try:
        for i, run_pointer in enumerate(collection.runs):
            current_status = query_run_status(run_pointer, iris_cache)
            state = states[i]

            if not should_dispatch(current_status, state.last_status, event_kind, state.consecutive_failures):
                state.last_status = current_status
                state.last_check = now
                continue

            outcome.dispatched += 1

            event = TickEvent(
                kind=event_kind,
                collection_name=collection_name,
                run_index=i,
                run_pointer=run_pointer,
                prompt=collection.prompt,
                logbook=collection.logbook,
                branch=collection.branch,
                issue=collection.issue,
                timestamp=now,
            )

            if worktree_path is None:
                worktree_path = setup_worktree(repo_root, collection.branch)

            result = agent.launch(event, worktree_path)
            _handle_result(result, event, worktree_path, state, outcome)

            state.last_status = current_status
            state.last_check = now

    finally:
        if worktree_path is not None:
            cleanup_worktree(repo_root, worktree_path)
        save_state(repo_root, collection_name, states)

    return outcome


def _handle_result(
    result: AgentResult,
    event: TickEvent,
    worktree_path: Path,
    state: RunState,
    outcome: TickOutcome,
) -> None:
    if result.success and result.logbook_entry:
        append_logbook(worktree_path, event.logbook, result.logbook_entry)
        pushed = commit_and_push(
            worktree_path,
            event.logbook,
            f"dispatch: update logbook for {event.collection_name} run {event.run_index}",
            event.branch,
        )
        if pushed:
            state.consecutive_failures = 0
            outcome.succeeded += 1
        else:
            logger.error("Failed to push logbook update for %s", event.collection_name)
            state.consecutive_failures += 1
            state.last_error = "push failed after merge retries"
            outcome.failed += 1
    else:
        state.consecutive_failures += 1
        state.last_error = result.error or "unknown error"
        outcome.failed += 1

    # Post issue comment: include failure context when the agent itself failed.
    if result.issue_comment:
        post_progress_comment(
            event.issue,
            result.issue_comment,
            event.collection_name,
            event.branch,
            event.logbook,
        )
    elif not result.success:
        error_detail = result.error or state.last_error or "unknown error"
        post_progress_comment(
            event.issue,
            f"Agent dispatch failed: {error_detail}",
            event.collection_name,
            event.branch,
            event.logbook,
        )

    if result.escalate or state.consecutive_failures >= ESCALATION_THRESHOLD:
        error = result.error or state.last_error or "repeated failures"
        post_escalation(event.issue, event.collection_name, error, event.branch)
        outcome.escalated += 1
