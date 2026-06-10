# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggressive Python execution tracer injected into untrusted SWE-rebench containers.

This module is bind-mounted into the sandboxed container at ``/_marin/tracer.py``
and loaded via ``PYTHONSTARTUP`` so it activates before pytest imports any user
code. It records one event per function call, return, and (optionally) line
execution and writes them as msgpack-framed records to the file descriptor
named in the ``MARIN_TRACE_FD`` environment variable.

Three goals:

1. **Project-only filtering**: trace only files under ``MARIN_TRACE_ROOTS``
   (colon-separated path prefixes; defaults to ``/testbed`` which is where
   SWE-rebench installs the project under test). stdlib, site-packages, and
   pytest internals are skipped — they make the trace 100x larger without
   adding signal.

2. **Bounded output**: emit at most ``MARIN_TRACE_MAX_EVENTS`` events
   (default 5,000,000). After the cap is hit the tracer disables itself and
   sets a flag the consumer can read.

3. **3.12 → sys.monitoring, 3.10/3.11 → sys.settrace**: PEP 669 monitoring
   has lower overhead and a richer event vocabulary on 3.12+. The fallback
   path uses sys.settrace and emits a similar event shape.

The output stream uses length-prefixed JSON records:

    [4-byte big-endian payload length][JSON payload]

where each payload is a dict like::

    {"e": "call", "f": "/testbed/foo.py", "l": 12, "n": "do_thing"}
    {"e": "return", "f": "/testbed/foo.py", "l": 18, "n": "do_thing"}
    {"e": "line", "f": "/testbed/foo.py", "l": 14, "n": "do_thing"}

A single ``{"e": "meta", ...}`` record is emitted at install time
carrying the actual tracer mode and Python version, so the host-side
reader doesn't have to guess.

The consumer (``run_one.py``) reads the captured stream from the
sandbox-side trace file and decodes the framed JSON records.

JSON is used instead of msgpack to avoid pulling msgpack into the
untrusted SWE-rebench images, which install their own pip environments.
The size cost is ~2x vs msgpack, which is fine for the prototype.
"""

from __future__ import annotations

import json
import os
import struct
import sys

_TRACE_ENABLED = False
_TRACE_FD: int = -1
_TRACE_ROOTS: tuple[str, ...] = ()
_MAX_EVENTS: int = 0
_EVENT_COUNT: int = 0
_TRUNCATED: bool = False
_TRACE_LINES: bool = False


def _emit(event: dict) -> None:
    """Write one length-prefixed JSON record to the trace fd."""
    global _EVENT_COUNT, _TRUNCATED, _TRACE_ENABLED
    if not _TRACE_ENABLED:
        return
    if _EVENT_COUNT >= _MAX_EVENTS:
        _TRUNCATED = True
        _TRACE_ENABLED = False
        _disable_tracer()
        return
    payload = json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header = struct.pack(">I", len(payload))
    try:
        os.write(_TRACE_FD, header + payload)
    except OSError:
        # The consumer closed the pipe — give up silently. The harness
        # already has whatever bytes it captured before close.
        _TRACE_ENABLED = False
        _disable_tracer()
        return
    _EVENT_COUNT += 1


def _path_in_roots(path: str) -> bool:
    if not path:
        return False
    for root in _TRACE_ROOTS:
        if path.startswith(root):
            return True
    return False


# ---------------------------------------------------------------------------
# sys.settrace path (Python 3.10 / 3.11)
# ---------------------------------------------------------------------------


def _settrace_callback(frame, event, arg):
    co = frame.f_code
    filename = co.co_filename
    if not _path_in_roots(filename):
        # Skip the entire frame: returning None means "don't trace this frame
        # or its children", which is exactly what we want for stdlib /
        # site-packages.
        return None
    if event == "call":
        _emit({"e": "call", "f": filename, "l": frame.f_lineno, "n": co.co_name})
        return _settrace_callback
    if event == "return":
        _emit({"e": "return", "f": filename, "l": frame.f_lineno, "n": co.co_name})
        return None
    if event == "exception":
        exc_type, exc_value, _tb = arg
        _emit(
            {
                "e": "exception",
                "f": filename,
                "l": frame.f_lineno,
                "n": co.co_name,
                "t": exc_type.__name__ if exc_type else "",
                "v": str(exc_value)[:200] if exc_value else "",
            }
        )
        return _settrace_callback
    if event == "line" and _TRACE_LINES:
        _emit({"e": "line", "f": filename, "l": frame.f_lineno, "n": co.co_name})
        return _settrace_callback
    return _settrace_callback


def _enable_settrace() -> None:
    sys.settrace(_settrace_callback)


def _disable_settrace() -> None:
    sys.settrace(None)


# ---------------------------------------------------------------------------
# sys.monitoring path (Python 3.12+)
# ---------------------------------------------------------------------------


_TOOL_ID = 4  # PROFILER slot is reserved by sys.monitoring; 4 is for user use.
_TOOL_NAME = "marin.swe_rebench_trace"


def _enable_monitoring() -> None:
    mon = sys.monitoring  # type: ignore[attr-defined]
    mon.use_tool_id(_TOOL_ID, _TOOL_NAME)
    events = mon.events
    event_mask = events.PY_START | events.PY_RETURN | events.RAISE
    if _TRACE_LINES:
        event_mask |= events.LINE
    mon.set_events(_TOOL_ID, event_mask)

    DISABLE = mon.DISABLE

    def on_py_start(code, instruction_offset):
        if not _path_in_roots(code.co_filename):
            return DISABLE
        _emit({"e": "call", "f": code.co_filename, "l": code.co_firstlineno, "n": code.co_name})
        return None

    def on_py_return(code, instruction_offset, retval):
        if not _path_in_roots(code.co_filename):
            return DISABLE
        _emit({"e": "return", "f": code.co_filename, "l": code.co_firstlineno, "n": code.co_name})
        return None

    def on_raise(code, instruction_offset, exception):
        if not _path_in_roots(code.co_filename):
            return DISABLE
        _emit(
            {
                "e": "exception",
                "f": code.co_filename,
                "l": code.co_firstlineno,
                "n": code.co_name,
                "t": type(exception).__name__,
                "v": str(exception)[:200],
            }
        )
        return None

    def on_line(code, line_number):
        if not _path_in_roots(code.co_filename):
            return DISABLE
        _emit({"e": "line", "f": code.co_filename, "l": line_number, "n": code.co_name})
        return None

    mon.register_callback(_TOOL_ID, events.PY_START, on_py_start)
    mon.register_callback(_TOOL_ID, events.PY_RETURN, on_py_return)
    mon.register_callback(_TOOL_ID, events.RAISE, on_raise)
    if _TRACE_LINES:
        mon.register_callback(_TOOL_ID, events.LINE, on_line)


def _disable_monitoring() -> None:
    mon = sys.monitoring  # type: ignore[attr-defined]
    try:
        mon.set_events(_TOOL_ID, 0)
        mon.free_tool_id(_TOOL_ID)
    except Exception:
        pass


def _disable_tracer() -> None:
    if sys.version_info >= (3, 12):
        _disable_monitoring()
    else:
        _disable_settrace()


def install() -> None:
    """Install the tracer based on environment configuration.

    Called from module-load time when the tracer is loaded via PYTHONSTARTUP.
    A no-op if MARIN_TRACE_FD is unset (so the module is safe to import in
    test contexts).
    """
    global _TRACE_ENABLED, _TRACE_FD, _TRACE_ROOTS, _MAX_EVENTS, _TRACE_LINES
    fd_str = os.environ.get("MARIN_TRACE_FD")
    if not fd_str:
        return
    try:
        _TRACE_FD = int(fd_str)
    except ValueError:
        return

    roots_env = os.environ.get("MARIN_TRACE_ROOTS", "/testbed")
    _TRACE_ROOTS = tuple(p for p in roots_env.split(os.pathsep) if p)

    try:
        _MAX_EVENTS = int(os.environ.get("MARIN_TRACE_MAX_EVENTS", "5000000"))
    except ValueError:
        _MAX_EVENTS = 5000000

    _TRACE_LINES = os.environ.get("MARIN_TRACE_LINES", "0") not in ("", "0", "false", "False")

    _TRACE_ENABLED = True
    tracer_mode = "sys.monitoring" if sys.version_info >= (3, 12) else "sys.settrace"
    if sys.version_info >= (3, 12):
        _enable_monitoring()
    else:
        _enable_settrace()

    # Emit a single metadata record at install time so the host-side reader
    # can recover the actual sandbox tracer mode and Python version, rather
    # than guessing from the worker's interpreter (which can differ).
    _emit(
        {
            "e": "meta",
            "tracer": tracer_mode,
            "py": f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}",
            "lines": _TRACE_LINES,
            "max_events": _MAX_EVENTS,
            "roots": list(_TRACE_ROOTS),
        }
    )


# Auto-install when imported via PYTHONSTARTUP. install() is idempotent and
# bails out if MARIN_TRACE_FD is not set, so importing this module from a
# unit test does nothing.
install()
