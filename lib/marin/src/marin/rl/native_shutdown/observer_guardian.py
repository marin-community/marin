# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone tracer subreaper; never imported/called inside a Ray actor."""
import argparse
import ctypes
import importlib
import json
import os
import resource
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
safety = importlib.import_module("stack_safety")
_MAX_READY_BYTES = 65536


def wait_for_debugger(config, expected, descriptor, owner_descriptor, started, result, *, now=time.monotonic):
    """Keep readiness startup separate from LLDB's ten-second ACK wait.

    GNU timeout still caps the complete debugger lifetime at fifteen seconds;
    these phase deadlines do not extend that outer process limit.
    """
    deadline = started + 9
    result.update(deadline_phase="startup_readiness", startup_deadline_monotonic=deadline)
    bound_ready = False
    ack = None
    marker_identity = {"pid": expected.pid, "starttime": expected.starttime, "nonce": config["nonce"]}
    while True:
        if owner_descriptor is not None and select.select([owner_descriptor], [], [], 0)[0]:
            raise RuntimeError("Observer owner exited")
        if config.get("cancel_path") and Path(config["cancel_path"]).exists():
            raise RuntimeError("Observer cancelled before completion")
        exited, _, _ = select.select([descriptor], [], [], 0.005)
        if exited:
            return
        observed = now()
        if observed > deadline:
            raise TimeoutError("Persistent tracer deadline exceeded")
        ready_path = Path(config["ready_path"])
        if not bound_ready and ready_path.exists():
            if ready_path.stat().st_size > _MAX_READY_BYTES:
                raise RuntimeError("Oversize observer readiness marker")
            ready = json.loads(ready_path.read_text())
            if ready != dict(marker_identity, running=True) or ready.get("running") is not True:
                raise RuntimeError("Observer readiness identity mismatch")
            observed = now()
            if observed > deadline:
                raise TimeoutError("Persistent tracer deadline exceeded")
            bound_ready = True
            deadline = observed + 10
            result.update(
                deadline_phase="await_ack",
                ready_observed_monotonic=observed,
                ack_wait_deadline_monotonic=deadline,
            )
        if bound_ready and ack is None and Path(config["ack_path"]).exists():
            ack = json.loads(Path(config["ack_path"]).read_text())
            if any(ack.get(key) != value for key, value in marker_identity.items()):
                raise RuntimeError("ACK identity mismatch")
            observed = now()
            if observed > deadline:
                raise TimeoutError("Persistent tracer deadline exceeded")
            deadline = ack["monotonic_ns"] / 1e9 + 2.95
            result.update(
                deadline_phase="post_ack",
                ack_observed_monotonic=observed,
                post_ack_deadline_monotonic=deadline,
            )
        if now() > deadline:
            raise TimeoutError("Persistent tracer deadline exceeded")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError("Could not become tracer subreaper")
    expected = safety.Identity(config["pid"], config["starttime"], config["pgrp"])
    binding = safety.LinuxBinding(expected)
    baseline = safety.direct_children()
    owner_descriptor = None
    owner_verified = "owner_pid" not in config
    debugger = None
    descriptor = None
    reader = None
    buffer = bytearray()
    seen = [0]
    result = {
        "status": "error",
        "traced_runtime_acceptance_qualified": False,
        "errors": [],
        "nonce": config["nonce"],
        "debugger_started": False,
        "debugger_reaped": True,
        "helpers_cleanup_established": True,
        "reader_stopped": True,
        "pid": expected.pid,
        "starttime": expected.starttime,
    }
    started = time.monotonic()

    def drain():
        while data := debugger.stdout.read(4096):
            seen[0] += len(data)
            buffer.extend(data[: max(0, 32768 - len(buffer))])

    try:
        if "owner_pid" in config:
            owner_descriptor = os.pidfd_open(config["owner_pid"], 0)
            if safety.identity(config["owner_pid"]).starttime != config["owner_starttime"]:
                raise RuntimeError("Owner identity changed")
            owner_verified = True
        command = [
            config["timeout"],
            "--signal=KILL",
            "15",
            config["debugger"],
            "--no-lldbinit",
            "--batch",
            "-o",
            "command script import " + json.dumps(str(Path(__file__).with_name("lldb_observer.py"))),
            "-o",
            "native-control " + json.dumps(str(args.config)),
        ]
        debugger = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={"PATH": os.defpath, "HOME": "/nonexistent", "LANG": "C", "LC_ALL": "C", "DEBUGINFOD_URLS": ""},
        )
        result.update(
            debugger_started=True, debugger_reaped=False, helpers_cleanup_established=False, reader_stopped=False
        )
        descriptor = os.pidfd_open(debugger.pid, 0)
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        wait_for_debugger(config, expected, descriptor, owner_descriptor, started, result)
    except Exception as error:
        result["errors"].append({"type": type(error).__name__, "message": str(error)[:400]})
    finally:
        if debugger is not None:
            # Reserve tracer numeric PID until owned debugger group is killed.
            try:
                os.killpg(debugger.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                debugger.wait(timeout=0.2)
                result["debugger_reaped"] = True
                result["debugger_exit_code"] = debugger.returncode
            except subprocess.TimeoutExpired:
                result["errors"].append({"type": "DebuggerReapTimeout"})
            try:
                result["adopted_helpers_reaped"] = safety.reap_debugger_descendants(
                    baseline, expected.pid, time.monotonic() + 0.2
                )
                result["helpers_cleanup_established"] = True
            except Exception as error:
                result["errors"].append({"type": "DescendantCleanupError", "message": str(error)[:200]})
        if descriptor is not None:
            os.close(descriptor)
        try:
            result["resume"] = binding.resume()
            # pidfd readiness, unlike a numeric /proc lookup, stays bound after
            # original Popen.wait reaps the child and the numeric PID is reused.
            gone, _, _ = select.select([binding.fd], [], [], 0)
            result["after"] = {"State": "original_target_exited", "TracerPid": "0"} if gone else binding.verify()
            if result["after"].get("TracerPid") != "0" or result["after"].get("State", "").startswith(("T", "t")):
                result["errors"].append({"type": "TargetStillTracedOrStopped"})
        except Exception as error:
            result["errors"].append({"type": "ResumeOrIdentityError", "message": str(error)[:200]})
        try:
            owner_unavailable = not owner_verified or (
                owner_descriptor is not None and bool(select.select([owner_descriptor], [], [], 0)[0])
            )
            if owner_unavailable:
                # The target pidfd was bound before owner setup. Never signal
                # the unavailable owner by numeric PID or accept a live orphan.
                binding.kill()
                result["orphan_target_kill_sent"] = True
                result["orphan_target_exit_observed"] = bool(select.select([binding.fd], [], [], 0.2)[0])
                if not result["orphan_target_exit_observed"]:
                    raise RuntimeError("Bound orphan target termination was not observed")
                result["after"] = {"State": "original_target_exited", "TracerPid": "0"}
        except Exception as error:
            result["errors"].append({"type": "OrphanTargetCleanupError", "message": str(error)[:200]})
        finally:
            if owner_descriptor is not None:
                os.close(owner_descriptor)
        binding.close()
        if reader is not None:
            reader.join(timeout=0.1)
            result["reader_stopped"] = not reader.is_alive()
            if reader.is_alive():
                result["errors"].append({"type": "ReaderStillAlive"})
        if debugger is not None and debugger.stdout is not None and (reader is None or not reader.is_alive()):
            debugger.stdout.close()
    result.update(
        output_bytes_seen=seen[0],
        output_truncated=seen[0] > len(buffer),
        guardian_before_output_writes_seconds=time.monotonic() - started,
    )
    Path(config["debugger_log_path"]).write_bytes(buffer)
    safe_types = {
        "DebuggerReapTimeout",
        "DescendantCleanupError",
        "TargetStillTracedOrStopped",
        "ResumeOrIdentityError",
        "ReaderStillAlive",
        "OrphanTargetCleanupError",
    }
    result["safe_cleanup"] = (
        result["debugger_reaped"]
        and result["helpers_cleanup_established"]
        and result["reader_stopped"]
        and not any(row["type"] in safe_types for row in result["errors"])
    )
    if result["output_truncated"]:
        result["errors"].append({"type": "DebuggerLogTruncated"})
    if Path(config["lldb_receipt_path"]).exists():
        result["lldb"] = json.loads(Path(config["lldb_receipt_path"]).read_text())
    if (
        not result["errors"]
        and not result.get("lldb", {}).get("errors")
        and result.get("lldb", {}).get("status") in {"collected", "target_exited_no_sample"}
    ):
        result["status"] = "completed"
    result["guardian_elapsed_before_final_receipt_write_seconds"] = time.monotonic() - started
    result["guardian_final_receipt_prepare_monotonic"] = time.monotonic()
    inner_started = result.get("lldb", {}).get("capture_started_monotonic")
    if inner_started is not None:
        result["capture_to_guardian_final_receipt_prepare_seconds"] = time.monotonic() - inner_started
    Path(config["guardian_receipt_path"]).write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
