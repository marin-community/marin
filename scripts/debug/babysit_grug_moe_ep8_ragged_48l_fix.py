#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Babysit the reconstructed 1e23 EP8 ragged MoE resume run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.debug.manage_grug_moe_ep8_ragged_48l_fix import (
    DEFAULT_INITIALIZE_FROM,
    ROOT,
    complete_checkpoint_paths,
    _load_state,
    _with_client,
)
from fray.v1.cluster.ray.deps import build_runtime_env_for_packages
from ray.job_submission import JobSubmissionClient

DEFAULT_STATE_PATH = ROOT / "scratch" / "grug_moe_ep8_ragged_48l_babysitter_state.json"
DEFAULT_EVENT_LOG_PATH = ROOT / "scratch" / "grug_moe_ep8_ragged_48l_babysitter_events.jsonl"
DEFAULT_PID_PATH = ROOT / "scratch" / "grug_moe_ep8_ragged_48l_babysitter.pid"
MANAGER_SCRIPT = ROOT / "scripts" / "debug" / "manage_grug_moe_ep8_ragged_48l_fix.py"
MAX_EVENT_STRING_CHARS = 8_000
REDACTED_SECRET = "<redacted>"
WANDB_RUN_RE = re.compile(r"https://wandb\.ai/\S+/dial_moe/runs/\S+")
CHECKPOINT_STEP_RE = re.compile(r"step-(\d+)")
SECRET_JSON_RE = re.compile(r'("(?:HF_TOKEN|WANDB_API_KEY)"\s*:\s*")[^"]+(")')
SECRET_ASSIGNMENT_RE = re.compile(r"((?:HF_TOKEN|WANDB_API_KEY)=)[^\s,'\"]+")
TERMINAL_FAILURE_STATUSES = {"FAILED", "STOPPED"}
TERMINAL_SUCCESS_STATUSES = {"SUCCEEDED"}
TRANSIENT_QUERY_PATTERNS = (
    "Authentication required",
    "Missing authentication token",
    "Connection aborted",
    "Remote end closed connection",
    "cannot listen to port",
    "Failed to get IPs for cluster",
    "timed out after 30 seconds",
    "Dashboard for cluster",
    "Failed to connect to Ray at address",
)
RECOVERABLE_LOG_PATTERNS = (
    "runtime_env setup failed",
    "No matching distribution found for kitoken==0.10.2",
    "OwnerDiedError",
    "dead node",
    "node death",
    "No accelerator found",
    "FAILED_PRECONDITION",
    "Device or resource busy",
    "coordination service",
    "JAX distributed service detected fatal errors",
    "leader task was preempted",
    "Failed to disconnect from coordination service",
    "Failed to send RPC to coordination service",
)
UNRECOVERABLE_LOG_PATTERNS = ("initialize_from must be a checkpoint path",)


@dataclass
class RaySnapshot:
    status: str
    message: str
    logs: str


@dataclass
class BabysitterState:
    started_at: str
    active_submission_id: str
    active_run_id: str
    initialize_from: str
    restart_count: int = 0
    last_checked_at: str | None = None
    last_status: str | None = None
    last_message_hash: str | None = None
    latest_checkpoint: str | None = None
    latest_checkpoint_step: int | None = None
    last_wandb_url: str | None = None
    last_recovery_reason: str | None = None
    recent_events: list[dict[str, Any]] = field(default_factory=list)


_stop_requested = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def status_name(status: Any) -> str:
    value = getattr(status, "value", status)
    text = str(value)
    if "." in text:
        return text.rsplit(".", maxsplit=1)[-1].upper()
    return text.upper()


def message_hash(message: str) -> str:
    return hashlib.sha256(message.encode()).hexdigest()[:16] if message else ""


def redact_sensitive_text(text: str) -> str:
    text = SECRET_JSON_RE.sub(rf"\1{REDACTED_SECRET}\2", text)
    return SECRET_ASSIGNMENT_RE.sub(rf"\1{REDACTED_SECRET}", text)


def event_safe(value: Any) -> Any:
    if isinstance(value, str):
        text = redact_sensitive_text(value)
        if len(text) <= MAX_EVENT_STRING_CHARS:
            return text
        return f"{text[:MAX_EVENT_STRING_CHARS]}...[truncated {len(text) - MAX_EVENT_STRING_CHARS} chars]"
    if isinstance(value, dict):
        return {key: event_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [event_safe(item) for item in value]
    return value


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = event_safe({"ts": utc_now(), **event})
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True))
        f.write("\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def save_babysitter_state(path: Path, state: BabysitterState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(asdict(state), f, indent=2, sort_keys=True)
        f.write("\n")


def load_babysitter_state(path: Path) -> BabysitterState:
    launch_state = _load_state()
    if not path.exists():
        return BabysitterState(
            started_at=utc_now(),
            active_submission_id=launch_state.submission_id,
            active_run_id=launch_state.run_id,
            initialize_from=launch_state.initialize_from,
        )

    with path.open() as f:
        payload = json.load(f)
    state = BabysitterState(**payload)
    if state.active_submission_id != launch_state.submission_id:
        state.active_submission_id = launch_state.submission_id
        state.active_run_id = launch_state.run_id
        state.initialize_from = launch_state.initialize_from
    return state


def record_recent_event(state: BabysitterState, event: dict[str, Any]) -> None:
    state.recent_events.append({"ts": utc_now(), **event})
    state.recent_events = state.recent_events[-20:]


def install_signal_handlers() -> None:
    def _request_stop(signum: int, _: Any) -> None:
        del signum
        global _stop_requested
        _stop_requested = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)


def existing_process_alive(pid_path: Path) -> bool:
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return pid != os.getpid()


def write_pid(pid_path: Path) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{os.getpid()}\n")


def remove_pid(pid_path: Path) -> None:
    if not pid_path.exists():
        return
    try:
        pid = int(pid_path.read_text().strip())
    except ValueError:
        pid = os.getpid()
    if pid == os.getpid():
        pid_path.unlink()


def ray_snapshot(submission_id: str, cluster_config: str, *, include_logs: bool) -> RaySnapshot:
    result: dict[str, Any] = {}

    def _collect(client: JobSubmissionClient) -> None:
        info = client.get_job_info(submission_id)
        status = status_name(info.status)
        result["status"] = status
        result["message"] = info.message or ""
        result["logs"] = (
            client.get_job_logs(submission_id)
            if include_logs or status in TERMINAL_FAILURE_STATUSES or status in TERMINAL_SUCCESS_STATUSES
            else ""
        )

    _with_client(cluster_config, _collect)
    return RaySnapshot(status=result["status"], message=result["message"], logs=result["logs"])


def run_manager_launch(initialize_from: str) -> None:
    result = subprocess.run(
        [sys.executable, str(MANAGER_SCRIPT), "launch", "--initialize-from", initialize_from],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    stdout = redact_sensitive_text(result.stdout)
    stderr = redact_sensitive_text(result.stderr)
    if stdout:
        print(stdout, end="", flush=True)
    if stderr:
        print(stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"manager launch failed with exit code {result.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )


def checkpoint_paths_for_run(run_id: str) -> list[str]:
    checkpoint_base = f"gs://marin-us-central2/grug/{run_id}-*/checkpoints"
    return complete_checkpoint_paths(checkpoint_base)


def checkpoint_step(path: str) -> int:
    match = CHECKPOINT_STEP_RE.search(path)
    if match is None:
        return -1
    return int(match.group(1))


def latest_checkpoint(run_id: str) -> tuple[str | None, int | None]:
    paths = checkpoint_paths_for_run(run_id)
    if not paths:
        return None, None
    best = max(paths, key=checkpoint_step)
    return best, checkpoint_step(best)


def initialize_step(path: str) -> int:
    return checkpoint_step(path)


def recovery_checkpoint(state: BabysitterState) -> str:
    candidates = [state.initialize_from]
    if state.latest_checkpoint is not None:
        candidates.append(state.latest_checkpoint)
    return max(candidates, key=initialize_step)


def runtime_env_find_links_ready() -> bool:
    req_path = Path(build_runtime_env_for_packages(extra=["tpu"])["pip"]["packages"])
    requirements = req_path.read_text()
    return (
        "kitoken==0.10.2" in requirements
        and "https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4" in requirements
    )


def extract_wandb_url(logs: str) -> str | None:
    matches = WANDB_RUN_RE.findall(logs)
    if not matches:
        return None
    return matches[-1].rstrip(".")


def recoverable_signal(snapshot: RaySnapshot) -> str | None:
    haystack = f"{snapshot.message}\n{snapshot.logs}"
    for pattern in RECOVERABLE_LOG_PATTERNS:
        if pattern.lower() in haystack.lower():
            return pattern
    return None


def unrecoverable_signal(snapshot: RaySnapshot) -> str | None:
    haystack = f"{snapshot.message}\n{snapshot.logs}"
    for pattern in UNRECOVERABLE_LOG_PATTERNS:
        if pattern.lower() in haystack.lower():
            return pattern
    return None


def should_retry_query(exc: Exception) -> bool:
    text = str(exc)
    return any(pattern in text for pattern in TRANSIENT_QUERY_PATTERNS)


def recover(args: argparse.Namespace, state: BabysitterState, reason: str, event_log_path: Path) -> None:
    if state.restart_count >= args.max_restarts:
        append_event(
            event_log_path,
            {
                "event": "max_restarts_reached",
                "restart_count": state.restart_count,
                "max_restarts": args.max_restarts,
                "reason": reason,
            },
        )
        raise SystemExit(2)

    if "kitoken" in reason.lower() or "runtime_env" in reason.lower():
        if not runtime_env_find_links_ready():
            append_event(event_log_path, {"event": "runtime_env_fix_missing", "reason": reason})
            raise SystemExit(2)

    checkpoint = recovery_checkpoint(state)
    append_event(
        event_log_path,
        {
            "event": "relaunching",
            "reason": reason,
            "initialize_from": checkpoint,
            "previous_submission_id": state.active_submission_id,
            "previous_run_id": state.active_run_id,
        },
    )
    run_manager_launch(checkpoint)
    launch_state = _load_state()
    state.restart_count += 1
    state.active_submission_id = launch_state.submission_id
    state.active_run_id = launch_state.run_id
    state.initialize_from = launch_state.initialize_from
    state.last_status = None
    state.last_message_hash = None
    state.latest_checkpoint = None
    state.latest_checkpoint_step = None
    state.last_wandb_url = None
    state.last_recovery_reason = reason
    record_recent_event(state, {"event": "relaunched", "reason": reason, "submission_id": launch_state.submission_id})


def check_once(args: argparse.Namespace, state: BabysitterState, state_path: Path, event_log_path: Path) -> str:
    launch_state = _load_state()
    state.active_submission_id = launch_state.submission_id
    state.active_run_id = launch_state.run_id
    state.initialize_from = launch_state.initialize_from

    include_logs = args.always_fetch_logs or state.last_status != "RUNNING"
    snapshot = ray_snapshot(state.active_submission_id, launch_state.cluster_config, include_logs=include_logs)
    state.last_checked_at = utc_now()
    state.last_status = snapshot.status
    state.last_message_hash = message_hash(snapshot.message)

    checkpoint, step = latest_checkpoint(state.active_run_id)
    state.latest_checkpoint = checkpoint
    state.latest_checkpoint_step = step

    wandb_url = extract_wandb_url(snapshot.logs)
    if wandb_url is not None:
        state.last_wandb_url = wandb_url

    event = {
        "event": "check",
        "submission_id": state.active_submission_id,
        "run_id": state.active_run_id,
        "status": snapshot.status,
        "latest_checkpoint": checkpoint,
        "latest_checkpoint_step": step,
        "wandb_url": state.last_wandb_url,
    }
    append_event(event_log_path, event)
    record_recent_event(state, event)

    if snapshot.status in TERMINAL_SUCCESS_STATUSES:
        append_event(event_log_path, {"event": "terminal_success", "submission_id": state.active_submission_id})
        save_babysitter_state(state_path, state)
        return "done"

    if snapshot.status in TERMINAL_FAILURE_STATUSES:
        unrecoverable_reason = unrecoverable_signal(snapshot)
        if unrecoverable_reason is not None:
            append_event(
                event_log_path,
                {
                    "event": "terminal_failure_unrecoverable",
                    "reason": unrecoverable_reason,
                    "status": snapshot.status,
                    "message": snapshot.message or snapshot.status,
                    "submission_id": state.active_submission_id,
                    "run_id": state.active_run_id,
                },
            )
            save_babysitter_state(state_path, state)
            return "done"

        reason = recoverable_signal(snapshot)
        if reason is None:
            append_event(
                event_log_path,
                {
                    "event": "terminal_failure_unclassified",
                    "status": snapshot.status,
                    "message": snapshot.message or snapshot.status,
                    "submission_id": state.active_submission_id,
                    "run_id": state.active_run_id,
                },
            )
            save_babysitter_state(state_path, state)
            return "done"
        recover(args, state, reason, event_log_path)
        save_babysitter_state(state_path, state)
        return "recovered"

    signal_name = recoverable_signal(snapshot)
    if signal_name is not None and snapshot.status not in {"RUNNING", "PENDING"}:
        recover(args, state, signal_name, event_log_path)
        save_babysitter_state(state_path, state)
        return "recovered"

    save_babysitter_state(state_path, state)
    return "continue"


def sleep_with_stop(interval: int) -> None:
    end = time.monotonic() + interval
    while not _stop_requested and time.monotonic() < end:
        time.sleep(min(30, max(0, end - time.monotonic())))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=570)
    parser.add_argument("--startup-interval", type=int, default=120)
    parser.add_argument("--max-restarts", type=int, default=6)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--event-log-path", type=Path, default=DEFAULT_EVENT_LOG_PATH)
    parser.add_argument("--pid-path", type=Path, default=DEFAULT_PID_PATH)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--always-fetch-logs", action="store_true")
    return parser


def main() -> None:
    args = parser().parse_args()
    install_signal_handlers()
    if existing_process_alive(args.pid_path):
        raise RuntimeError(f"Babysitter already appears to be running: {args.pid_path}")
    write_pid(args.pid_path)

    state = load_babysitter_state(args.state_path)
    if state.initialize_from == "":
        state.initialize_from = DEFAULT_INITIALIZE_FROM

    try:
        append_event(
            args.event_log_path,
            {
                "event": "started",
                "submission_id": state.active_submission_id,
                "run_id": state.active_run_id,
                "restart_count": state.restart_count,
            },
        )
        use_startup_interval = state.last_checked_at is None
        while not _stop_requested:
            try:
                outcome = check_once(args, state, args.state_path, args.event_log_path)
            except Exception as exc:
                if should_retry_query(exc):
                    append_event(args.event_log_path, {"event": "transient_query_error", "error": str(exc)})
                    outcome = "continue"
                else:
                    append_event(args.event_log_path, {"event": "unhandled_error", "error": repr(exc)})
                    raise

            if args.once or outcome == "done":
                return
            if outcome == "recovered":
                next_interval = args.startup_interval
            elif use_startup_interval:
                next_interval = args.startup_interval
                use_startup_interval = False
            else:
                next_interval = args.interval
            sleep_with_stop(next_interval)
    finally:
        append_event(args.event_log_path, {"event": "stopped", "submission_id": state.active_submission_id})
        save_babysitter_state(args.state_path, state)
        remove_pid(args.pid_path)


if __name__ == "__main__":
    main()
