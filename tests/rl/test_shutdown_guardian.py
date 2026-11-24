# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Portable guardian phase timing using real marker files and readiness pipes."""

import importlib.util
import itertools
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
            IsInlined=lambda: False,
            GetPC=lambda: pc,
            GetModule=lambda: SimpleNamespace(
                IsValid=lambda: True, GetFileSpec=lambda: SimpleNamespace(GetFilename=lambda: "libnative.so")
            ),
            GetFunctionName=lambda: symbol,
        )

    def thread(thread_id, symbols):
        values = [
            frame(symbol, 0x1000 + (thread_id - 1212) * 0x100 + 16 * index) for index, symbol in enumerate(symbols)
        ]
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
            frames=values,
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
        frame=frame,
        thread=thread,
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


def test_lldb_deep_main_exposes_caller_above_eight_without_deepening_supplementary_threads(lldb_capture):
    value = lldb_capture
    value.main.frames[:] = [value.frame("cuda_driver", 0x2000 + index) for index in range(12)]
    value.main.frames.append(value.frame("owning_resource_destructor", 0x3000))
    value.other.frames[:] = [value.frame(f"helper_{index}", 0x4000 + index) for index in range(12)]
    value.run()
    data = Path(value.settings["trace_path"]).read_text()
    receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
    assert "frame #12: 0x3000 libnative.so owning_resource_destructor" in data
    assert "helper_7\n" in data and "helper_8\n" not in data
    assert receipt["main_frames_captured"] == 13 and receipt["main_frame_limit"] == 64
    assert receipt["main_stack_complete"] and receipt["main_stack_truncated"] is False
    assert not receipt["main_frame_limit_reached"]
    assert receipt["max_supplementary_frames_per_thread"] == 8
    assert receipt["threads_seen"] == receipt["threads_captured"] == 2
    assert receipt["native_stack_qualified"] and receipt["frames_complete"]


@pytest.mark.parametrize("main_depth", [64, 65])
def test_lldb_main_cap_requires_actual_end_of_stack_before_qualification(lldb_capture, main_depth):
    value = lldb_capture
    value.main.frames[:] = [value.frame(f"caller_{index}", 0x2000 + index) for index in range(main_depth)]
    value.run()
    data = Path(value.settings["trace_path"]).read_text()
    receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
    assert receipt["main_frames_captured"] == 64 and receipt["main_frame_limit_reached"]
    assert "caller_63\n" in data and "caller_64\n" not in data
    if main_depth == 64:
        assert receipt["main_stack_complete"] and receipt["main_stack_truncated"] is False
        assert receipt["native_stack_qualified"] and receipt["frames_complete"]
    else:
        assert not receipt["main_stack_complete"] and receipt["main_stack_truncated"] is True
        assert not receipt["native_stack_qualified"] and not receipt["frames_complete"]
        assert receipt["error_phase"] == "main_stack_limit_probe" and receipt["final_detach_success"]


@pytest.mark.parametrize("expired_frame", [10, 64])
def test_lldb_deep_main_deadline_retains_progress_and_never_qualifies_partial_stack(lldb_capture, expired_frame):
    value = lldb_capture
    value.main.frames[:] = [value.frame(f"caller_{index}", 0x2000 + index) for index in range(65)]
    observed = []

    def expire(index):
        if index == expired_frame:
            observed.append(Path(value.settings["trace_path"]).read_bytes())
            value.clock[0] = 2.8

    value.main.hook.before_frame = expire
    value.run()
    data = Path(value.settings["trace_path"]).read_bytes()
    receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
    assert data == observed[0] and b"caller_9\n" in data
    assert receipt["main_frames_captured"] == expired_frame
    assert not receipt["main_stack_complete"] and receipt["main_stack_truncated"] is None
    assert not receipt["native_stack_qualified"] and not receipt["frames_complete"]
    assert receipt["active_frame_index"] == expired_frame
    assert receipt["main_frame_limit_reached"] == (expired_frame == 64)
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
        IsInlined=lambda: False,
        GetPC=lambda: 0x1234,
        GetModule=lambda: SimpleNamespace(
            IsValid=lambda: True, GetFileSpec=lambda: SimpleNamespace(GetFilename=lambda: "native.so")
        ),
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


def test_lldb_repeated_noninline_PC_enriches_once_per_stop_but_writes_every_real_frame(lldb_capture):
    value = lldb_capture
    calls = {"pc": 0, "module": 0, "symbol": 0}

    def current_frame():
        frame = value.frame("late_cleanup", 0x1234)

        def pc():
            calls["pc"] += 1
            return 0x1234

        def module():
            calls["module"] += 1
            assert Path(value.settings["trace_path"]).read_bytes().endswith(b"0x1234 ")
            # Repeating the external symbol lookup would exceed the real
            # detach reserve; reusing enrichment must preserve full coverage.
            value.clock[0] += 0.2
            return SimpleNamespace(
                IsValid=lambda: True, GetFileSpec=lambda: SimpleNamespace(GetFilename=lambda: "native.so")
            )

        def symbol():
            calls["symbol"] += 1
            return "late_cleanup"

        frame.GetPC = pc
        frame.GetModule = module
        frame.GetFunctionName = symbol
        return frame

    value.main.frames[:] = [current_frame() for _ in range(3)]
    value.other.frames[:] = [current_frame() for _ in range(2)]
    for stop in range(2):
        if stop:
            Path(value.settings["trace_path"]).unlink()
        value.run()
        lines = Path(value.settings["trace_path"]).read_text().splitlines()
        receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
        assert [line for line in lines if line.startswith("thread ")] == [
            "thread tid=1212 main=true",
            "thread tid=1213 main=false",
        ]
        assert [line for line in lines if line.startswith("frame #")] == [
            f"frame #{index}: 0x1234 native.so late_cleanup" for index in (0, 1, 2, 0, 1)
        ]
        assert calls == {"pc": 5 * (stop + 1), "module": stop + 1, "symbol": stop + 1}
        assert receipt["status"] == "collected" and receipt["frames_written"] == 5
        assert receipt["capture_cost"]["cache_hits"] == 4 and receipt["capture_cost"]["cache_misses"] == 1
        assert receipt["capture_cost"]["cache_entries"] == 1 and receipt["capture_cost"]["enrichments"]["calls"] == 1
        assert receipt["capture_cost"]["frame_unwinds"]["calls"] == 7  # Five frames and two real invalid probes.


def test_lldb_inline_frames_keep_distinct_context_even_when_noninline_PC_is_cached(lldb_capture):
    value = lldb_capture
    value.main.frames[:] = [value.frame("noninline", 0x1234)]
    value.other.frames[:] = [value.frame("inline_A", 0x1234), value.frame("inline_B", 0x1234)]
    for frame in value.other.frames:
        frame.IsInlined = lambda: True
    value.run()
    data = Path(value.settings["trace_path"]).read_text()
    receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
    assert all(name in data for name in ("noninline", "inline_A", "inline_B"))
    assert receipt["frames_written"] == 3 and receipt["capture_cost"]["enrichments"]["calls"] == 3
    assert receipt["capture_cost"]["inline_bypasses"] == 2 and receipt["capture_cost"]["cache_hits"] == 0
    assert receipt["capture_cost"]["cache_entries"] == 1


@pytest.mark.parametrize("unknown", ["invalid_module", "unknown_module", "invalid_symbol", "unknown_symbol"])
def test_lldb_unknown_enrichment_is_not_memoized_and_can_resolve_later(lldb_capture, unknown):
    value = lldb_capture
    lookups = []
    frames = []
    for index in range(3):
        frame = value.frame("resolved", 0x1234)

        def module(index=index):
            lookups.append(index)
            return SimpleNamespace(
                IsValid=lambda: not (index == 0 and unknown == "invalid_module"),
                GetFileSpec=lambda: SimpleNamespace(
                    GetFilename=lambda: (
                        None
                        if index == 0 and unknown == "unknown_module"
                        else "invalid_module.so" if index == 0 and unknown == "invalid_module" else "native.so"
                    )
                ),
            )

        frame.GetModule = module
        if index == 0 and unknown == "invalid_symbol":
            frame.GetFunctionName = lambda: None
            frame.GetSymbol = lambda: SimpleNamespace(IsValid=lambda: False, GetName=lambda: "invalid_lookup")
        elif index == 0 and unknown == "unknown_symbol":
            frame.GetFunctionName = lambda: "<unknown>"
        frames.append(frame)
    value.main.frames[:] = frames
    value.run()
    data = Path(value.settings["trace_path"]).read_text()
    assert lookups == [0, 1]  # Only the valid second resolution is reused by frame three.
    assert "frame #1: 0x1234 native.so resolved" in data and "frame #2: 0x1234 native.so resolved" in data
    assert json.loads(Path(value.settings["lldb_receipt_path"]).read_text())["status"] == "collected"


@pytest.mark.parametrize("blocked", [False, True])
def test_lldb_supplementary_journals_preserve_complete_or_faulted_real_stream(lldb_capture, monkeypatch, blocked):
    value = lldb_capture
    checkpoints = []
    partial = []
    original = value.module.os.replace

    def record(source, target):
        original(source, target)
        if str(target) == value.settings["lldb_receipt_path"]:
            checkpoints.append((value.clock[0], json.loads(Path(target).read_text())))

    monkeypatch.setattr(value.module.os, "replace", record)
    for index in range(100):
        thread = value.thread(1300 + index, ["real_helper"])

        def advance(frame_index, thread_index=index):
            value.clock[0] += 0.001
            if blocked and thread_index == 90 and frame_index == 1:
                partial.append(
                    (
                        Path(value.settings["trace_path"]).read_bytes(),
                        json.loads(Path(value.settings["lldb_receipt_path"]).read_text()),
                        value.clock[0],
                    )
                )
                raise TimeoutError("Later supplementary unwind blocked")

        thread.hook.before_frame = advance
        value.threads.append(thread)
    value.run()
    rows = [row for _, row in checkpoints]
    periodic = [
        (at, row)
        for at, row in checkpoints
        if row["phase"] in {"thread_lookup", "thread_frames", "thread_complete"} and row.get("active_thread_id") != 1212
    ]
    assert 1 <= len(periodic) <= 3
    assert all(b[0] - a[0] >= 0.1 for a, b in itertools.pairwise(periodic))
    assert all(
        phase in [row["phase"] for row in rows] for phase in ("main_thread", "thread_inventory", "supplementary_threads")
    )
    receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
    lines = Path(value.settings["trace_path"]).read_text().splitlines()
    if blocked:
        assert sum(line.startswith("thread ") for line in lines) == 93 and receipt["threads_captured"] == 92
        assert sum(line.startswith("frame #") for line in lines) == receipt["frames_written"] == 95
        assert Path(value.settings["trace_path"]).read_bytes() == partial[0][0]
        assert 0 <= partial[0][2] - partial[0][1]["journal_monotonic"] <= 0.101
        assert partial[0][1]["phase"] in {"thread_lookup", "thread_frames", "thread_complete"}
        assert receipt["error_phase"] == "thread_frames" and receipt["active_thread_id"] == 1390
        assert receipt["status"] == "error" and not receipt["frames_complete"] and not receipt["native_stack_qualified"]
        assert receipt["final_detach_success"]
    else:
        assert sum(line.startswith("thread ") for line in lines) == receipt["threads_captured"] == 102
        assert sum(line.startswith("frame #") for line in lines) == receipt["frames_written"] == 104
        assert receipt["status"] == "collected" and receipt["frames_complete"]
    assert receipt["capture_cost"]["frame_unwinds"]["seconds"] > 0


def test_lldb_names_cannot_forge_extra_streamed_thread_or_frame_records(lldb_capture):
    value = lldb_capture
    frame = value.main.frames[0]
    frame.GetModule = lambda: SimpleNamespace(
        IsValid=lambda: True, GetFileSpec=lambda: SimpleNamespace(GetFilename=lambda: "native\nthread tid=999.so")
    )
    frame.GetFunctionName = lambda: "late_cleanup\r\nframe #999: forged"
    value.run()
    lines = Path(value.settings["trace_path"]).read_text().splitlines()
    receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
    assert sum(line.startswith("thread ") for line in lines) == receipt["threads_captured"] == 2
    assert sum(line.startswith("frame #") for line in lines) == receipt["frames_written"] == 4


def test_lldb_capture_cost_counters_are_bounded_aggregate_completed_operation_times(lldb_capture, monkeypatch):
    value = lldb_capture
    original_replace = value.module.os.replace
    original_fdopen = value.module.os.fdopen

    def replace(source, target):
        value.clock[0] += 0.001
        original_replace(source, target)

    def fdopen(*args, **kwargs):
        stream = original_fdopen(*args, **kwargs)

        def write(data):
            value.clock[0] += 0.0001
            return stream.write(data)

        return SimpleNamespace(write=write, close=stream.close)

    monkeypatch.setattr(value.module.os, "replace", replace)
    monkeypatch.setattr(value.module.os, "fdopen", fdopen)
    for thread in value.threads:
        thread.hook.before_frame = lambda index: value.clock.__setitem__(0, value.clock[0] + 0.0002)
    value.run()
    receipt = json.loads(Path(value.settings["lldb_receipt_path"]).read_text())
    costs = receipt["capture_cost"]
    assert receipt["status"] == "collected"
    assert all(
        costs[name]["calls"] > 0 and costs[name]["seconds"] > 0
        for name in ("journal_writes", "frame_unwinds", "enrichments", "trace_writes")
    )
    assert costs["frame_unwinds"]["calls"] == 6 and costs["enrichments"]["calls"] == 4
    assert costs["journal_writes"]["seconds"] == pytest.approx(costs["journal_writes"]["calls"] * 0.001)
    assert costs["trace_writes"]["seconds"] == pytest.approx(costs["trace_writes"]["calls"] * 0.0001)
    assert costs["frame_unwinds"]["seconds"] == pytest.approx(0.0012)
