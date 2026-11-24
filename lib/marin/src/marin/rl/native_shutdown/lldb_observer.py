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

LIMIT = 1024 * 1024
_MAX_FRAMES_PER_THREAD = 8
_MAX_MAIN_FRAMES = 64
_DETACH_RESERVE_SECONDS = 0.2
_THREAD_JOURNAL_INTERVAL_SECONDS = 0.1


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def run(debugger, command, result, internal_dict):
    settings = json.loads(Path(shlex.split(command)[0]).read_text())
    costs: dict = {
        name: {"calls": 0, "seconds": 0.0} for name in ("journal_writes", "frame_unwinds", "enrichments", "trace_writes")
    }
    costs.update(cache_hits=0, cache_misses=0, inline_bypasses=0, cache_entries=0)
    receipt: dict = {
        "status": "error",
        "native_stack_qualified": False,
        "post_ACK_numeric_attach_count": 0,
        "traced_runtime_acceptance_qualified": False,
        "phase": "settings",
        "errors": [],
        "frames_complete": False,
        "main_thread_id": settings["pid"],
        "threads_seen": 0,
        "threads_captured": 0,
        "frames_written": 0,
        "max_supplementary_frames_per_thread": _MAX_FRAMES_PER_THREAD,
        "main_frame_limit": _MAX_MAIN_FRAMES,
        "main_frames_captured": 0,
        "main_frame_limit_reached": False,
        "main_stack_complete": False,
        "main_stack_truncated": None,
        "output_bytes_seen": 0,
        "trace_bytes_written": 0,
        "output_truncated": False,
        "late_cleanup_frame": False,
        "libxla_frame_present": False,
        "frame_marker_present": False,
        "capture_cost": costs,
        "capture_cost_scope": (
            "Non-exhaustive capture-phase elapsed times; enrichment includes module-prefix trace writes, "
            "so timings overlap. Attach/ACK wait and final receipt write are excluded."
        ),
    }
    process = None
    trace = None
    last_journal = None
    # Addresses are stable only for this single stopped-process sample. Inline
    # frames can share a PC while carrying different call contexts.
    enrichment_by_pc = {}
    measuring_capture = False

    def measured(kind, function, *args):
        if not measuring_capture:
            return function(*args)
        started = time.monotonic()
        costs[kind]["calls"] += 1
        try:
            return function(*args)
        finally:
            costs[kind]["seconds"] += time.monotonic() - started

    def journal(phase, *, periodic=False, **fields):
        nonlocal last_journal
        now = time.monotonic()
        receipt.update(phase=phase, journal_monotonic=now, **fields)
        if not periodic or last_journal is None or now - last_journal >= _THREAD_JOURNAL_INTERVAL_SECONDS:
            measured("journal_writes", atomic_json, settings["lldb_receipt_path"], receipt)
            last_journal = now

    def execute(text):
        response = lldb.SBCommandReturnObject()
        debugger.GetCommandInterpreter().HandleCommand(text, response)
        if not response.Succeeded():
            raise RuntimeError((response.GetError() or "LLDB command failed")[:300])

    def write_trace(text):
        data = text.encode()
        receipt["output_bytes_seen"] += len(data)
        if receipt["output_bytes_seen"] > LIMIT:
            receipt["output_truncated"] = True
            raise RuntimeError("OutputLimitExceeded")
        assert trace is not None
        written = measured("trace_writes", trace.write, data)
        receipt["trace_bytes_written"] += written or 0
        if written != len(data):
            raise OSError("Incomplete native stack write")

    def capture_budget(deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError("Native stack capture reached its detach reserve")

    def capture_thread(thread, deadline, *, main):
        thread_id = thread.GetThreadID()
        write_trace(f"thread tid={thread_id} main={str(main).lower()}\n")
        captured = 0
        if not main:
            # The streamed frame prefix retains the per-frame cursor. Avoid
            # thousands of atomic JSON rewrites while the target is stopped.
            receipt.pop("active_frame_index", None)
            journal("thread_frames", periodic=True, active_thread_id=thread_id)
        frame_limit = _MAX_MAIN_FRAMES if main else _MAX_FRAMES_PER_THREAD
        for index in range(frame_limit):
            capture_budget(deadline)
            if main:
                journal("frame_unwind", active_thread_id=thread_id, active_frame_index=index)
            frame = measured("frame_unwinds", thread.GetFrameAtIndex, index)
            capture_budget(deadline)
            if not frame.IsValid():
                if main:
                    receipt.update(main_stack_complete=True, main_stack_truncated=False)
                break
            pc = frame.GetPC()
            if pc == lldb.LLDB_INVALID_ADDRESS:
                raise RuntimeError("Stopped frame has no program counter")
            # FileIO is unbuffered: even blocked module/symbol resolution or
            # SIGKILL leaves the numeric prefix, without a qualifying verdict.
            write_trace(f"frame #{index}: {pc:#x} ")
            if main:
                journal("frame_symbol", active_thread_id=thread_id, active_frame_index=index)
            inline = frame.IsInlined()
            capture_budget(deadline)
            cached = None if inline else enrichment_by_pc.get(pc)
            if cached is None:
                costs["inline_bypasses" if inline else "cache_misses"] += 1
                enrichment_started = time.monotonic()
                costs["enrichments"]["calls"] += 1
                try:
                    resolved_module = frame.GetModule()
                    module = resolved_module.GetFileSpec().GetFilename() or "<unknown>"
                    capture_budget(deadline)
                    module = module.replace("\n", " ").replace("\r", " ")
                    write_trace(module + " ")
                    symbol = frame.GetFunctionName()
                    capture_budget(deadline)
                    symbol_valid = bool(symbol and symbol.strip() and symbol != "<unknown>")
                    if not symbol:
                        resolved_symbol = frame.GetSymbol()
                        symbol = resolved_symbol.GetName()
                        capture_budget(deadline)
                        symbol_valid = bool(
                            resolved_symbol.IsValid() and symbol and symbol.strip() and symbol != "<unknown>"
                        )
                    symbol = (symbol or "<unknown>").replace("\n", " ").replace("\r", " ")
                    if not inline and resolved_module.IsValid() and module != "<unknown>" and symbol_valid:
                        enrichment_by_pc[pc] = (module, symbol)
                        costs["cache_entries"] = len(enrichment_by_pc)
                finally:
                    costs["enrichments"]["seconds"] += time.monotonic() - enrichment_started
            else:
                costs["cache_hits"] += 1
                module, symbol = cached
                write_trace(module + " ")
            write_trace(symbol + "\n")
            receipt["late_cleanup_frame"] |= "late_cleanup" in symbol
            receipt["libxla_frame_present"] |= any(name in module for name in ("libjax", "libxla", "xla_extension"))
            captured += 1
            receipt["frames_written"] += 1
            receipt["frame_marker_present"] = True
            if main:
                receipt["main_frames_captured"] = captured
                receipt["main_frame_limit_reached"] = captured == frame_limit
        if main and receipt["main_frame_limit_reached"]:
            # One validity-only probe distinguishes an exact-cap stack from a
            # truncated caller chain. Never print or enrich the extra frame.
            journal("main_stack_limit_probe", active_thread_id=thread_id, active_frame_index=frame_limit)
            capture_budget(deadline)
            frame = measured("frame_unwinds", thread.GetFrameAtIndex, frame_limit)
            capture_budget(deadline)
            receipt["main_stack_truncated"] = frame.IsValid()
            if receipt["main_stack_truncated"]:
                raise RuntimeError("Main thread stack exceeds its frame limit")
            receipt["main_stack_complete"] = True
        if not captured:
            raise RuntimeError("Stopped thread has no readable frames")
        receipt["threads_captured"] += 1
        journal("thread_complete", periodic=not main, active_thread_id=thread_id, completed_thread_frames=captured)

    try:
        descriptor = os.open(settings["trace_path"], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        trace = os.fdopen(descriptor, "wb", buffering=0)
        journal("settings")
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
        journal("pre_ACK_attach")
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
        journal("pre_ACK_continue")
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
        journal("await_binary_ACK_marker")
        deadline = time.monotonic() + 10
        while not Path(settings["ack_path"]).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("No fixture ACK marker")
            time.sleep(0.005)
        ack = json.loads(Path(settings["ack_path"]).read_text())
        if ack["pid"] != settings["pid"] or ack["nonce"] != settings["nonce"]:
            raise RuntimeError("ACK identity mismatch")
        receipt["ack"] = ack
        journal("await_capture")
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
        measuring_capture = True
        journal("capture", capture_started_monotonic=started)
        if process.GetState() == lldb.eStateExited:
            receipt.update(status="target_exited_no_sample", phase="done", target_exit_status=process.GetExitStatus())
            return
        journal("stop_requested")
        error = process.Stop()
        if not error.Success():
            raise RuntimeError("Stop failed: " + str(error))
        deadline = time.monotonic() + 0.25
        while process.GetState() != lldb.eStateStopped and time.monotonic() < deadline:
            time.sleep(0.002)
        if process.GetState() != lldb.eStateStopped:
            raise RuntimeError("Target did not stop for stack")
        # The external guardian still ends at ACK+2.95. Reserve detach time
        # inside that window and the original inner one-second bound.
        deadline = min(started + 1, ack["monotonic_ns"] / 1e9 + 2.95) - _DETACH_RESERVE_SECONDS
        journal("main_thread")
        capture_budget(deadline)
        main = process.GetThreadByID(settings["pid"])
        capture_budget(deadline)
        if not main.IsValid() or main.GetThreadID() != settings["pid"]:
            raise RuntimeError("Bound Linux main thread is unavailable")
        capture_thread(main, deadline, main=True)
        journal("thread_inventory")
        count = process.GetNumThreads()
        capture_budget(deadline)
        receipt["threads_seen"] = count
        receipt.pop("active_frame_index", None)
        journal("supplementary_threads", active_thread_id=None, active_thread_index=0)
        seen = set()
        for index in range(count):
            capture_budget(deadline)
            journal("thread_lookup", periodic=True, active_thread_index=index)
            thread = process.GetThreadAtIndex(index)
            capture_budget(deadline)
            thread_id = thread.GetThreadID()
            if not thread.IsValid() or thread_id in seen:
                raise RuntimeError("Stopped thread inventory is incomplete")
            seen.add(thread_id)
            if thread_id != settings["pid"]:
                capture_thread(thread, deadline, main=False)
        if receipt["threads_captured"] != count:
            raise RuntimeError("Stopped thread coverage differs")
        capture_budget(deadline)
        receipt["frames_complete"] = True
        journal("detach")
        error = process.Detach(False)
        if not error.Success():
            raise RuntimeError("Detach failed: " + str(error))
        receipt["stop_backtrace_detach_seconds"] = time.monotonic() - started
        if receipt["stop_backtrace_detach_seconds"] > 1:
            raise TimeoutError("Inner Stop/backtrace/Detach exceeded1second")
        receipt.update(status="collected", phase="done", native_stack_qualified=True)
    except Exception as error:
        receipt["error_phase"] = receipt["phase"]
        receipt["errors"].append({"type": type(error).__name__, "message": str(error)[:400]})
        result.SetError(str(error)[:400])
    finally:
        if (
            process is not None
            and process.IsValid()
            and process.GetState() not in (lldb.eStateExited, lldb.eStateDetached)
        ):
            journal("final_detach")
            error = process.Detach(False)
            receipt["final_detach_success"] = error.Success()
            if not error.Success():
                receipt["errors"].append({"type": "FinalDetachError", "message": str(error)[:200]})
        if receipt["errors"]:
            receipt["status"] = "error"
            receipt["native_stack_qualified"] = False
        if trace is not None:
            trace.close()
        receipt["stack_write_completed_monotonic"] = time.monotonic()
        receipt["LLDB_receipt_prepare_monotonic"] = time.monotonic()
        atomic_json(settings["lldb_receipt_path"], receipt)


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand("command script add -f lldb_observer.run native-control")
