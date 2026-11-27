# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in parent-owned native shutdown observation; no changes to process transport."""

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

from marin.rl.native_shutdown.stack_safety import identity

_ASSETS = Path(__file__).with_name("native_shutdown")
_MAX_RECEIPT_BYTES = 65536
_UNSAFE_ERRORS = {
    "DebuggerReapTimeout",
    "DescendantCleanupError",
    "ReaderStillAlive",
    "TargetStillTracedOrStopped",
    "ResumeOrIdentityError",
    "OrphanTargetCleanupError",
}


def validated_ack_marker(payload: dict) -> dict | None:
    """Validate a fresh, private marker destination before cleanup begins."""
    marker = payload.get("diagnostic_shutdown_observer")
    if marker is None:
        return None
    if set(marker) != {"pid", "starttime", "nonce", "ack_path"}:
        raise ValueError("Invalid shutdown observer marker fields")
    if marker["pid"] != os.getpid() or marker["starttime"] != identity(os.getpid()).starttime:
        raise ValueError("Shutdown observer target identity differs")
    if len(marker["nonce"]) != 32 or uuid.UUID(hex=marker["nonce"]).hex != marker["nonce"]:
        raise ValueError("Invalid shutdown observer nonce")
    path = Path(marker["ack_path"])
    directory = path.parent.stat()
    if (
        not path.is_absolute()
        or path.name != "ack.json"
        or path.exists()
        or not stat.S_ISDIR(directory.st_mode)
        or directory.st_uid != os.getuid()
        or stat.S_IMODE(directory.st_mode) != 0o700
    ):
        raise ValueError("Shutdown marker requires a fresh private local directory")
    return dict(marker)


def publish_ack_marker(marker: dict, *, process_id: int | None) -> None:
    """Publish only after write_frame flushed the real binary acknowledgement."""
    row = {
        **marker,
        "event": "shutdown_ack_written",
        "process_id": process_id,
        "time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
    }
    path = Path(marker["ack_path"])
    temporary = path.with_name("ack.pending")
    data = (json.dumps(row) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        # link is atomic and cannot overwrite an earlier observation.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ShutdownObserver:
    """Own one bounded guardian while the original synchronous request executes."""

    def __init__(self, child: subprocess.Popen, destination: Path, process_id: int):
        if sys.platform != "linux" or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
            raise RuntimeError("Shutdown observation requires Linux and protocol-owned child reaping")
        timeout = shutil.which("timeout")
        if timeout is None or not Path("/usr/bin/lldb").is_file():
            raise RuntimeError("Shutdown observation requires GNU timeout and /usr/bin/lldb")
        target = identity(child.pid)
        owner = identity(os.getpid())
        nonce = uuid.uuid4().hex
        self.directory = destination / "native-shutdown-observer" / f"rank{process_id}-pid{target.pid}-{nonce}"
        self.directory.mkdir(parents=True, mode=0o700)
        # Output volumes can add inherited setgid despite mkdir's requested mode.
        self.directory.chmod(0o700, follow_symlinks=False)
        self.config = {
            "pid": target.pid,
            "starttime": target.starttime,
            "pgrp": target.pgrp,
            "nonce": nonce,
            "owner_pid": owner.pid,
            "owner_starttime": owner.starttime,
            "debugger": "/usr/bin/lldb",
            "timeout": timeout,
        }
        for key, name in (
            ("ready_path", "tracer-ready.json"),
            ("trace_path", "stack.txt"),
            ("lldb_receipt_path", "lldb-receipt.json"),
            ("guardian_receipt_path", "guardian-receipt.json"),
            ("debugger_log_path", "debugger.log"),
            ("ack_path", "ack.json"),
            ("cancel_path", "cancel"),
        ):
            self.config[key] = str(self.directory / name)
        config_path = self.directory / "observer-config.json"
        config_path.write_text(json.dumps(self.config) + "\n")
        self.retirement_attempted = False
        self.artifact_handoff_attempted = False
        self.stderr = (self.directory / "guardian.stderr").open("wb")
        try:
            self.guardian = subprocess.Popen(
                [sys.executable, "-I", str(_ASSETS / "observer_guardian.py"), "--config", str(config_path)],
                stdout=subprocess.DEVNULL,
                stderr=self.stderr,
                start_new_session=True,
            )
        except BaseException:
            self.stderr.close()
            raise

    def wait_ready(self) -> None:
        deadline = time.monotonic() + 9
        ready = Path(self.config["ready_path"])
        while not ready.exists():
            if self.guardian.poll() is not None:
                raise RuntimeError("Shutdown guardian exited before readiness")
            if time.monotonic() > deadline:
                raise TimeoutError("Shutdown guardian readiness timed out")
            time.sleep(0.005)
        expected = {key: self.config[key] for key in ("pid", "starttime", "nonce")}
        if ready.stat().st_size > _MAX_RECEIPT_BYTES or json.loads(ready.read_text()) != dict(expected, running=True):
            raise RuntimeError("Shutdown guardian readiness binding differs")

    def payload(self) -> dict:
        if self.retirement_attempted:
            raise RuntimeError("Observer teardown unverified; shutdown/restart is blocked")
        return {
            "diagnostic_shutdown_observer": {key: self.config[key] for key in ("pid", "starttime", "nonce", "ack_path")}
        }

    def retire(self, *, cancel: bool = False) -> dict:
        """Prove target/tracer/helper cleanup before allowing the owner to close/restart."""
        self.retirement_attempted = True
        if cancel:
            Path(self.config["cancel_path"]).touch(exist_ok=True)
        self.guardian.wait(timeout=2)
        path = Path(self.config["guardian_receipt_path"])
        if path.stat().st_size > _MAX_RECEIPT_BYTES:
            raise RuntimeError("Oversize shutdown guardian receipt")
        row = json.loads(path.read_text())
        after = row.get("after", {})
        errors = row.get("errors")
        safe = (
            all(row.get(key) == self.config[key] for key in ("pid", "starttime", "nonce"))
            and row.get("safe_cleanup") is True
            and row.get("debugger_reaped") is True
            and row.get("helpers_cleanup_established") is True
            and row.get("reader_stopped") is True
            and isinstance(errors, list)
            and all(
                isinstance(e, dict) and isinstance(e.get("type"), str) and e["type"] not in _UNSAFE_ERRORS
                for e in errors
            )
            and after.get("TracerPid") == "0"
            and isinstance(after.get("State"), str)
            and not after["State"].startswith(("T", "t"))
        )
        if not safe:
            raise RuntimeError("Shutdown observer cleanup unknown/unsafe; child reservation retained")
        self.stderr.close()
        return row


def observation_completed(row: dict) -> bool:
    """A module list, truncated trace or observer error never qualifies collection."""
    trace = row.get("lldb", {})
    return (
        row.get("status") == "completed"
        and row.get("errors") == []
        and trace.get("errors") == []
        and trace.get("pre_ACK_numeric_attach_count") == 1
        and trace.get("post_ACK_numeric_attach_count") == 0
        and not trace.get("output_truncated", False)
        and (
            trace.get("status") == "target_exited_no_sample"
            or (trace.get("status") == "collected" and trace.get("native_stack_qualified") is True)
        )
    )
