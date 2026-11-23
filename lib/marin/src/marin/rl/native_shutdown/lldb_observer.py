# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit trusted LLDB command using public SB APIs; no target locals/arguments."""
import json
import os
import shlex
import time
from pathlib import Path

import lldb  # pyrefly: ignore[missing-import]  # supplied by the LLDB interpreter
from stack_safety import identity  # pyrefly: ignore[missing-import]  # standalone sibling asset

LIMIT = 128 * 1024


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def run(debugger, command, result, internal_dict):
    settings = json.loads(Path(shlex.split(command)[0]).read_text())
    receipt: dict = {
        "status": "error",
        "native_stack_qualified": False,
        "post_ACK_numeric_attach_count": 0,
        "traced_runtime_acceptance_qualified": False,
        "phase": "settings",
        "errors": [],
    }
    process = None
    trace = bytearray()

    def execute(text):
        response = lldb.SBCommandReturnObject()
        debugger.GetCommandInterpreter().HandleCommand(text, response)
        if not response.Succeeded():
            raise RuntimeError((response.GetError() or "LLDB command failed")[:300])
        return (response.GetOutput() or "").encode()

    try:
        for text in [
            "settings set auto-confirm true",
            "settings set use-color false",
            "settings set target.load-script-from-symbol-file false",
            'settings set frame-format "frame #${frame.index}: ${frame.pc} ${module.file.basename} ${function.name}\\n"',
            "settings set stop-line-count-before 0",
            "settings set stop-line-count-after 0",
        ]:
            execute(text)
        if identity(settings["pid"]).starttime != settings["starttime"]:
            raise RuntimeError("Pre-attach identity changed")
        receipt["phase"] = "pre_ACK_attach"
        debugger.SetAsync(False)
        target = debugger.CreateTarget("")
        error = lldb.SBError()
        process = target.AttachToProcessWithID(debugger.GetListener(), settings["pid"], error)
        if not error.Success() or process.GetProcessID() != settings["pid"]:
            raise RuntimeError("Attach failed: " + str(error))
        if identity(settings["pid"]).starttime != settings["starttime"]:
            raise RuntimeError("Attached identity changed")
        receipt.update(
            bound_pid=process.GetProcessID(),
            LLDB_process_unique_id=process.GetUniqueID(),
            pre_ACK_numeric_attach_count=1,
        )
        debugger.SetAsync(True)
        error = process.Continue()
        if not error.Success():
            raise RuntimeError("Continue failed: " + str(error))
        deadline = time.monotonic() + 0.3
        while process.GetState() != lldb.eStateRunning and time.monotonic() < deadline:
            time.sleep(0.002)
        if process.GetState() != lldb.eStateRunning:
            raise RuntimeError("Not running after pre-ACK attach")
        atomic_json(
            settings["ready_path"],
            {"pid": settings["pid"], "starttime": settings["starttime"], "nonce": settings["nonce"], "running": True},
        )
        receipt["phase"] = "await_binary_ACK_marker"
        deadline = time.monotonic() + 10
        while not Path(settings["ack_path"]).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("No fixture ACK marker")
            time.sleep(0.005)
        ack = json.loads(Path(settings["ack_path"]).read_text())
        if ack["pid"] != settings["pid"] or ack["nonce"] != settings["nonce"]:
            raise RuntimeError("ACK identity mismatch")
        receipt["ack"] = ack
        if settings.get("fault") == "hang":
            time.sleep(20)
        if settings.get("fault") == "crash":
            os.kill(os.getpid(), 9)
        sample_at = ack["monotonic_ns"] / 1e9 + 2
        while time.monotonic() < sample_at:
            if process.GetState() == lldb.eStateExited:
                receipt.update(
                    status="target_exited_no_sample", phase="done", target_exit_status=process.GetExitStatus()
                )
                return
            time.sleep(0.005)
        started = time.monotonic()
        receipt.update(phase="capture", capture_started_monotonic=started)
        if process.GetState() == lldb.eStateExited:
            receipt.update(status="target_exited_no_sample", phase="done")
            return
        error = process.Stop()
        if not error.Success():
            raise RuntimeError("Stop failed: " + str(error))
        deadline = time.monotonic() + 0.25
        while process.GetState() != lldb.eStateStopped and time.monotonic() < deadline:
            time.sleep(0.002)
        if process.GetState() != lldb.eStateStopped:
            raise RuntimeError("Target did not stop for stack")
        stack = execute("thread backtrace -c 8 all")
        images = execute("image list -o -f")
        combined = stack + b"\n=== MODULES (not frames) ===\n" + images
        trace.extend(combined[:LIMIT])
        receipt.update(output_bytes_seen=len(combined), output_truncated=len(combined) > LIMIT)
        if len(combined) > LIMIT:
            raise RuntimeError("OutputLimitExceeded")
        receipt["late_cleanup_frame"] = "late_cleanup" in stack.decode(errors="replace")
        receipt["libxla_frame_present"] = any(name in stack for name in [b"libjax", b"libxla", b"xla_extension"])
        receipt["frame_marker_present"] = b"frame #0" in stack
        error = process.Detach(False)
        if not error.Success():
            raise RuntimeError("Detach failed: " + str(error))
        receipt["stop_backtrace_detach_seconds"] = time.monotonic() - started
        if receipt["stop_backtrace_detach_seconds"] > 1:
            raise TimeoutError("Inner Stop/backtrace/Detach exceeded1second")
        receipt.update(status="collected", phase="done", native_stack_qualified=receipt["frame_marker_present"])
    except Exception as error:
        receipt["errors"].append({"type": type(error).__name__, "message": str(error)[:400]})
        result.SetError(str(error)[:400])
    finally:
        if (
            process is not None
            and process.IsValid()
            and process.GetState() not in (lldb.eStateExited, lldb.eStateDetached)
        ):
            error = process.Detach(False)
            receipt["final_detach_success"] = error.Success()
            if not error.Success():
                receipt["errors"].append({"type": "FinalDetachError", "message": str(error)[:200]})
        if receipt["errors"]:
            receipt["status"] = "error"
            receipt["native_stack_qualified"] = False
        Path(settings["trace_path"]).write_bytes(trace)
        receipt["stack_write_completed_monotonic"] = time.monotonic()
        receipt["LLDB_receipt_prepare_monotonic"] = time.monotonic()
        atomic_json(settings["lldb_receipt_path"], receipt)


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand("command script add -f lldb_observer.run native-control")
