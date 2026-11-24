# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Portable guardian phase timing using real marker files and readiness pipes."""

import importlib.util
import json
import os
import select
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from marin.rl.native_shutdown.observer_guardian import safety, wait_for_debugger


@pytest.fixture
def observer_wait(tmp_path):
    read, write = os.pipe()
    config = {"nonce": "c" * 32, "ready_path": str(tmp_path / "ready.json"), "ack_path": str(tmp_path / "ack.json")}
    expected = safety.Identity(pid=41, starttime=73, pgrp=41)
    try:
        yield config, expected, read, write
    finally:
        os.close(read)
        os.close(write)


def advance_on_poll(monkeypatch, times, publish=None):
    clock = [0]
    ticks = iter(times)
    real_select = select.select

    def poll(readers, writers, errors, timeout):
        if timeout:
            clock[0] = next(ticks)
            if publish is not None:
                publish(clock[0])
        return real_select(readers, writers, errors, 0)

    monkeypatch.setattr(select, "select", poll)
    return clock


def publish_marker(config, expected, kind, monotonic):
    marker = {"pid": expected.pid, "starttime": expected.starttime, "nonce": config["nonce"]}
    path = Path(config[f"{kind}_path"])
    pending = path.with_suffix(".pending")
    pending.write_text(
        json.dumps(dict(marker, running=True) if kind == "ready" else dict(marker, monotonic_ns=int(monotonic * 1e9)))
    )
    if kind == "ready":
        os.replace(pending, path)
    else:
        os.link(pending, path)
        pending.unlink()


def test_cold_readiness_leaves_a_new_ack_window_and_keeps_post_ack_deadline(observer_wait, monkeypatch):
    config, expected, descriptor, finished = observer_wait

    def publish(value):
        if value == 8.25:
            publish_marker(config, expected, "ready", value)
        elif value == 8.5:
            publish_marker(config, expected, "ack", value)
        elif value == 10.75:
            os.write(finished, b"exited")

    clock = advance_on_poll(monkeypatch, [7.75, 8.25, 8.5, 10.75], publish)
    result = {}
    wait_for_debugger(config, expected, descriptor, None, 0, result, now=lambda: clock[0])
    assert result == {
        "deadline_phase": "post_ack",
        "startup_deadline_monotonic": 9,
        "ready_observed_monotonic": 8.25,
        "ack_wait_deadline_monotonic": 18.25,
        "ack_observed_monotonic": 8.5,
        "post_ack_deadline_monotonic": 11.45,
    }


@pytest.mark.parametrize(
    "phase,times", [("startup_readiness", [9.1]), ("await_ack", [1, 11.1]), ("post_ack", [1, 3.96])]
)
def test_each_guardian_phase_still_rejects_its_expired_budget(observer_wait, phase, times, monkeypatch):
    config, expected, descriptor, _ = observer_wait
    marker = {"pid": expected.pid, "starttime": expected.starttime, "nonce": config["nonce"]}
    if phase != "startup_readiness":
        Path(config["ready_path"]).write_text(json.dumps(dict(marker, running=True)))
    if phase == "post_ack":
        Path(config["ack_path"]).write_text(json.dumps(dict(marker, monotonic_ns=1_000_000_000)))
    clock = advance_on_poll(monkeypatch, times)
    result = {}
    with pytest.raises(TimeoutError, match="Persistent tracer deadline exceeded"):
        wait_for_debugger(config, expected, descriptor, None, 0, result, now=lambda: clock[0])
    assert result["deadline_phase"] == phase


def test_simultaneously_observed_ready_and_ack_keep_the_actual_ack_deadline(observer_wait, monkeypatch):
    config, expected, descriptor, finished = observer_wait

    def publish(value):
        if value == 8.5:
            publish_marker(config, expected, "ready", value)
            publish_marker(config, expected, "ack", value)
        else:
            os.write(finished, b"exited")

    clock = advance_on_poll(monkeypatch, [8.5, 10.75], publish)
    result = {}
    wait_for_debugger(config, expected, descriptor, None, 0, result, now=lambda: clock[0])
    assert result["ready_observed_monotonic"] == result["ack_observed_monotonic"] == 8.5
    assert result["post_ack_deadline_monotonic"] == 11.45


@pytest.mark.parametrize(
    "kind,times,read_finishes_at,phase",
    [
        ("ready", [9.1], None, "startup_readiness"),
        ("ack", [1, 11.1], None, "await_ack"),
        ("both", [9.1], None, "startup_readiness"),
        ("ready", [8.9], 9.1, "startup_readiness"),
        ("ack", [1, 10.9], 11.1, "await_ack"),
    ],
)
def test_late_markers_cannot_rearm_an_expired_phase(observer_wait, monkeypatch, kind, times, read_finishes_at, phase):
    config, expected, descriptor, _ = observer_wait

    def publish(value):
        if kind == "ack" and value == 1:
            publish_marker(config, expected, "ready", value)
        elif kind == "both":
            publish_marker(config, expected, "ready", value)
            publish_marker(config, expected, "ack", value)
        else:
            publish_marker(config, expected, kind, value)

    clock = advance_on_poll(monkeypatch, times, publish)
    if read_finishes_at is not None:
        marker_path = Path(config[f"{kind}_path"])
        real_read = Path.read_text

        def delayed_read(path, *args, **kwargs):
            value = real_read(path, *args, **kwargs)
            if path == marker_path:
                clock[0] = read_finishes_at
            return value

        monkeypatch.setattr(Path, "read_text", delayed_read)
    result = {}
    with pytest.raises(TimeoutError, match="Persistent tracer deadline exceeded"):
        wait_for_debugger(config, expected, descriptor, None, 0, result, now=lambda: clock[0])
    assert result["deadline_phase"] == phase
    assert "post_ack_deadline_monotonic" not in result


@pytest.mark.parametrize("field", ["pid", "starttime", "nonce"])
@pytest.mark.parametrize("marker_kind", ["ready", "ack"])
def test_guardian_rejects_markers_bound_to_a_different_target(observer_wait, field, marker_kind):
    config, expected, descriptor, _ = observer_wait
    marker = {"pid": expected.pid, "starttime": expected.starttime, "nonce": config["nonce"]}
    changed = dict(marker)
    changed[field] = "different" if field == "nonce" else int(changed[field]) + 1
    Path(config["ready_path"]).write_text(json.dumps(dict(changed if marker_kind == "ready" else marker, running=True)))
    if marker_kind == "ack":
        Path(config["ack_path"]).write_text(json.dumps(dict(changed, monotonic_ns=1_000_000_000)))
    with pytest.raises(RuntimeError, match="identity mismatch"):
        wait_for_debugger(config, expected, descriptor, None, 0, {}, now=lambda: 1)


@pytest.fixture
def lldb_capture(tmp_path, monkeypatch):
    # LLDB is an external process/API boundary. Its portable fake exposes the
    # same stopped threads while allowing unwind and symbol faults on demand.
    clock = [2.0]
    settings = {
        "pid": 1212,
        "starttime": 73,
        "nonce": "f" * 32,
        "ready_path": str(tmp_path / "ready.json"),
        "ack_path": str(tmp_path / "ack.json"),
        "trace_path": str(tmp_path / "stack.txt"),
        "lldb_receipt_path": str(tmp_path / "lldb-receipt.json"),
    }
    Path(settings["ack_path"]).write_text(json.dumps({"pid": 1212, "nonce": "f" * 32, "monotonic_ns": 0}))
    config = tmp_path / "observer.json"
    config.write_text(json.dumps(settings))
    state = SimpleNamespace(value=1, stop_fault=None)
    invalid = SimpleNamespace(IsValid=lambda: False)

    def frame(symbol, pc):
        return SimpleNamespace(
            IsValid=lambda: True,
            GetPC=lambda: pc,
            GetModule=lambda: SimpleNamespace(GetFileSpec=lambda: SimpleNamespace(GetFilename=lambda: "libnative.so")),
            GetFunctionName=lambda: symbol,
        )

    def thread(thread_id, symbols):
        values = [frame(symbol, 0x1000 + 16 * index) for index, symbol in enumerate(symbols)]
        hook = SimpleNamespace(before_frame=None)

        def get_frame(index):
            if hook.before_frame is not None:
                hook.before_frame(index)
            return values[index] if index < len(values) else invalid

        return SimpleNamespace(
            IsValid=lambda: True,
            GetThreadID=lambda: thread_id,
            GetFrameAtIndex=get_frame,
            hook=hook,
        )

    main = thread(1212, ["late_cleanup", "Py_Finalize"])
    other = thread(1213, ["helper_wait", "native_thread_main"])
    # The main thread deliberately is not first in the debugger inventory.
    threads = [other, main]

    def success():
        return SimpleNamespace(Success=lambda: True)

    def continue_process():
        state.value = 1
        return success()

    def stop_process():
        if state.stop_fault is not None:
            state.stop_fault()
        state.value = 2
        return success()

    def detach_process(keep_stopped):
        state.value = 4
        return success()

    process = SimpleNamespace(
        IsValid=lambda: True,
        GetProcessID=lambda: 1212,
        GetUniqueID=lambda: 99,
        GetState=lambda: state.value,
        Continue=continue_process,
        Stop=stop_process,
        Detach=detach_process,
        GetThreadByID=lambda thread_id: next((v for v in threads if v.GetThreadID() == thread_id), invalid),
        GetNumThreads=lambda: len(threads),
        GetThreadAtIndex=lambda index: threads[index],
    )

    def interpret(command, result):
        result.error = None
        result.output = ""
        if command.startswith("thread backtrace"):
            # Also model the old buffered command so the regression is the
            # missing durable output, rather than a fake rejecting old argv.
            for value in threads:
                for index in range(8):
                    current = value.GetFrameAtIndex(index)
                    if not current.IsValid():
                        break
                    result.output += f"frame #{index}: {current.GetFunctionName()}\n"
        elif command == "image list -o -f":
            result.output = "native modules\n"
        elif not command.startswith("settings set "):
            raise AssertionError("Unexpected LLDB command")

    def command_result():
        value = SimpleNamespace(error=None, output="")
        value.Succeeded = lambda: value.error is None
        value.GetError = lambda: value.error
        value.GetOutput = lambda: value.output
        return value

    debugger = SimpleNamespace(
        GetCommandInterpreter=lambda: SimpleNamespace(HandleCommand=interpret),
        SetAsync=lambda value: None,
        CreateTarget=lambda path: SimpleNamespace(AttachToProcessWithID=lambda *args: process),
        GetListener=lambda: None,
    )
    provider = SimpleNamespace(
        SBCommandReturnObject=command_result,
        SBError=success,
        eStateRunning=1,
        eStateStopped=2,
        eStateExited=3,
        eStateDetached=4,
        LLDB_INVALID_ADDRESS=2**64 - 1,
    )
    monkeypatch.setitem(sys.modules, "lldb", provider)
    monkeypatch.setitem(sys.modules, "stack_safety", SimpleNamespace(identity=lambda pid: SimpleNamespace(starttime=73)))
    source = Path(safety.__file__).with_name("lldb_observer.py")
    spec = importlib.util.spec_from_file_location("portable_lldb_capture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda seconds: None))
    errors = []
    result = SimpleNamespace(SetError=errors.append)
    return SimpleNamespace(
        module=module,
        run=lambda: module.run(debugger, json.dumps(str(config)), result, None),
        settings=settings,
        main=main,
        other=other,
        threads=threads,
        state=state,
        clock=clock,
        errors=errors,
    )


def test_lldb_capture_streams_main_before_other_threads_and_reports_complete_coverage(lldb_capture):
    value = lldb_capture
    value.run()
    data = Path(value.settings["trace_path"]).read_bytes()
    receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
    assert data.startswith(b"thread tid=1212 main=true\n")
    assert all(name in data for name in (b"late_cleanup", b"Py_Finalize", b"helper_wait", b"native_thread_main"))
    assert receipt["status"] == "collected" and receipt["errors"] == []
    assert receipt["native_stack_qualified"] and receipt["frames_complete"]
    assert receipt["threads_seen"] == receipt["threads_captured"] == 2
    assert receipt["frames_written"] == 4 and receipt["late_cleanup_frame"]
    assert receipt["trace_bytes_written"] == len(data) == receipt["output_bytes_seen"]
    assert receipt["pre_ACK_numeric_attach_count"] == 1 and receipt["post_ACK_numeric_attach_count"] == 0
    assert value.state.value == 4 and value.errors == []


def test_lldb_later_unwind_failure_preserves_already_written_main_frames(lldb_capture):
    value = lldb_capture
    observed = []

    def blocked(index):
        if index == 1:
            path = Path(value.settings["trace_path"])
            observed.append(path.read_bytes() if path.exists() else None)
            raise TimeoutError("Unwinder blocked after an earlier frame")

    value.other.hook.before_frame = blocked
    value.run()
    receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
    data = Path(value.settings["trace_path"]).read_bytes()
    assert observed and observed[0] is not None and b"late_cleanup" in observed[0]
    assert data == observed[0]  # Finally must not replace the durable partial file.
    assert receipt["status"] == "error" and receipt["error_phase"] == "thread_frames"
    assert receipt["active_thread_id"] == 1213 and "active_frame_index" not in receipt
    assert receipt["frames_written"] == 3 and receipt["threads_captured"] == 1
    assert not receipt["frames_complete"] and not receipt["native_stack_qualified"]
    assert receipt["final_detach_success"] and value.state.value == 4


@pytest.mark.parametrize("operation", ["GetFrameAtIndex", "GetModule", "GetFunctionName"])
def test_lldb_late_frame_return_cannot_qualify_capture_after_detach_reserve(lldb_capture, operation):
    value = lldb_capture
    target = value.other if operation == "GetFrameAtIndex" else value.other.GetFrameAtIndex(0)
    original = getattr(target, operation)

    def slow(*args):
        value.clock[0] = 2.8
        return original(*args)

    setattr(target, operation, slow)
    value.run()
    receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
    assert b"late_cleanup" in Path(value.settings["trace_path"]).read_bytes()
    assert receipt["status"] == "error" and receipt["error_phase"] == "thread_frames"
    assert "active_frame_index" not in receipt
    assert not receipt["frames_complete"] and not receipt["native_stack_qualified"]
    assert receipt["final_detach_success"]


def test_lldb_oversize_symbol_preserves_bounded_nonqualifying_partial_output(lldb_capture):
    value = lldb_capture
    oversized = SimpleNamespace(
        IsValid=lambda: True,
        GetPC=lambda: 0x1234,
        GetModule=lambda: SimpleNamespace(GetFileSpec=lambda: SimpleNamespace(GetFilename=lambda: "native.so")),
        GetFunctionName=lambda: "x" * (1024 * 1024 + 1),
    )
    value.main.GetFrameAtIndex = lambda index: oversized
    value.run()
    receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
    data = Path(value.settings["trace_path"]).read_bytes()
    assert data.startswith(b"thread tid=1212 main=true\nframe #0: 0x1234 native.so ")
    assert len(data) <= 1024 * 1024 and receipt["output_truncated"]
    assert receipt["status"] == "error" and receipt["error_phase"] == "frame_symbol"
    assert not receipt["frames_complete"] and not receipt["native_stack_qualified"]
    assert receipt["final_detach_success"]


def test_lldb_existing_partial_trace_is_not_overwritten_on_reentry(lldb_capture):
    value = lldb_capture
    path = Path(value.settings["trace_path"])
    path.write_bytes(b"partial trace retained before a failed reentry")
    value.run()
    assert path.read_bytes() == b"partial trace retained before a failed reentry"
    receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
    assert receipt["status"] == "error" and receipt["errors"][0]["type"] == "FileExistsError"
    assert not receipt["frames_complete"] and not receipt["native_stack_qualified"]


def test_lldb_stop_failure_leaves_phase_journal_without_qualifying_frames(lldb_capture):
    value = lldb_capture
    observed = []

    def blocked():
        observed.append(json.loads(Path(value.settings["lldb_receipt_path"]).read_text()))
        raise TimeoutError("Stop blocked")

    value.state.stop_fault = blocked
    value.run()
    receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
    assert observed[0]["phase"] == "stop_requested" and observed[0]["status"] == "error"
    assert Path(value.settings["trace_path"]).read_bytes() == b""
    assert receipt["error_phase"] == "stop_requested" and receipt["final_detach_success"]
    assert not receipt["frames_complete"] and not receipt["native_stack_qualified"]


@pytest.mark.parametrize("operation", ["GetModule", "GetFunctionName"])
def test_lldb_blocked_frame_enrichment_keeps_numeric_prefix_nonqualifying(lldb_capture, operation):
    value = lldb_capture
    first = value.main.GetFrameAtIndex(0)
    observed = []

    def blocked():
        observed.append(Path(value.settings["trace_path"]).read_bytes())
        raise TimeoutError("Frame enrichment blocked")

    setattr(first, operation, blocked)
    value.run()
    data = Path(value.settings["trace_path"]).read_bytes()
    receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
    assert data == observed[0] and data.startswith(b"thread tid=1212 main=true\nframe #0: 0x1000 ")
    assert receipt["error_phase"] == "frame_symbol" and receipt["frames_written"] == 0
    assert receipt["status"] == "error" and not receipt["frames_complete"]
    assert not receipt["native_stack_qualified"] and receipt["final_detach_success"]
