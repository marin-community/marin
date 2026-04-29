# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming execution tracer, auto-injected via PYTHONPATH.

Put the directory containing this file on PYTHONPATH and Python will import
sitecustomize at startup, which registers a sys.settrace callback that writes
an annotated-source view (real source lines + locals-diff comments) to
TRACER_OUTPUT as the traced program runs.

Configure via environment variables:

    TRACER_OUTPUT           - JSONL path for per-test rows (default: /tmp/trace.jsonl)
    TRACER_REPO_ROOT        - repo root (whitelist filter + relative display paths; default: cwd)
    TRACER_PER_TEST_EVENTS  - per-test event budget before shrinking call-stack depth (default: 10000)
"""

import atexit
import dataclasses
import inspect
import io
import json
import linecache
import os
import sys
import threading

_SKIP_MARKERS = (
    "site-packages",
    "lib/python",
    "/usr/lib",
    "importlib",
    "<frozen",
    "/pytracer/",
    "/.local/",
    "/usr/local/lib/",
    "_pytest/",
    "pluggy/",
    "conftest.py",
    "_distutils_hack",
    "pkg_resources",
)

PER_TEST_BUDGET = int(os.environ.get("TRACER_PER_TEST_EVENTS", "10000"))

MAX_RECUR_DEPTH = 5
MAX_CONTAINER_ITEMS = 10
MAX_REPR_LEN = 1000


def _type_name(obj):
    try:
        return type(obj).__name__
    except Exception:
        return "?"


def _safe_repr(obj):
    """Render obj, shrinking recursion depth progressively if the rendered
    string overflows MAX_REPR_LEN. At the extreme, a value that cannot be
    rendered within the budget collapses to `<TypeName>`.
    """
    try:
        for max_depth in range(MAX_RECUR_DEPTH, -1, -1):
            s = _render(obj, 0, max_depth)
            if len(s) <= MAX_REPR_LEN:
                return s
        return f"<{_type_name(obj)}>"
    except Exception:
        return f"<{_type_name(obj)}>"


def _render(obj, depth, max_depth):
    """Inner renderer. Uses <TypeName> as the placeholder beyond max_depth
    and as the leaf-overflow fallback — no string truncation, never leaks
    exceptions (the catch-all returns `<TypeName>`)."""
    if depth > max_depth:
        return f"<{_type_name(obj)}>"

    try:
        if obj is None or isinstance(obj, (bool, int, float, complex)):
            return repr(obj)
        if isinstance(obj, (str, bytes, bytearray)):
            r = repr(obj)
            # Long strings can't shrink via depth — collapse to a length tag.
            return r if len(r) <= MAX_REPR_LEN else f"<{_type_name(obj)} len={len(obj)}>"
    except Exception:
        return f"<{_type_name(obj)}>"

    if isinstance(obj, dict):
        items = list(obj.items())[:MAX_CONTAINER_ITEMS]
        inner = ", ".join(f"{_render(k, depth + 1, max_depth)}: {_render(v, depth + 1, max_depth)}" for k, v in items)
        if len(obj) > MAX_CONTAINER_ITEMS:
            inner += ", ..."
        return "{" + inner + "}"
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = list(obj)[:MAX_CONTAINER_ITEMS]
        inner = ", ".join(_render(x, depth + 1, max_depth) for x in items)
        if len(obj) > MAX_CONTAINER_ITEMS:
            inner += ", ..."
        if isinstance(obj, list):
            return f"[{inner}]"
        if isinstance(obj, tuple):
            return f"({inner}{',' if len(obj) == 1 else ''})"
        if isinstance(obj, set):
            return "{" + inner + "}" if obj else "set()"
        return f"frozenset({{{inner}}})"

    cls = type(obj)
    mod = getattr(cls, "__module__", "") or ""
    name = cls.__name__

    if mod.startswith(("numpy", "jax", "torch", "tensorflow")) and hasattr(obj, "shape"):
        shape = getattr(obj, "shape", None)
        dtype = getattr(obj, "dtype", None)
        device = getattr(obj, "device", None)
        parts = [f"{mod.split('.')[0]}.{name}", f"shape={tuple(shape) if shape is not None else '?'}"]
        if dtype is not None:
            parts.append(f"dtype={dtype}")
        if device is not None:
            parts.append(f"device={device}")
        return "<" + " ".join(parts) + ">"

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return f"{name}({_render(dataclasses.asdict(obj), depth + 1, max_depth)})"
        except Exception:
            pass

    try:
        r = repr(obj)
    except Exception:
        return f"<{name}>"
    if " object at 0x" in r and hasattr(obj, "__dict__"):
        try:
            return f"{name}({_render(vars(obj), depth + 1, max_depth)})"
        except Exception:
            pass
    # Long leaf reprs can't shrink by depth either — collapse to type.
    return r if len(r) <= MAX_REPR_LEN else f"<{name}>"


def _should_skip(filename, repo_root):
    if not filename or filename.startswith("<"):
        return True
    if repo_root:
        try:
            resolved = os.path.realpath(filename)
        except Exception:
            return True
        if not resolved.startswith(repo_root + os.sep) and resolved != repo_root:
            return True
    for marker in _SKIP_MARKERS:
        if marker in filename:
            return True
    return False


def _snapshot_locals(frame):
    out = {}
    for k, v in frame.f_locals.items():
        if k.startswith("__") and k.endswith("__"):
            continue
        if k.startswith(("@py_assert", "@pytest_ar")):
            continue
        out[k] = _safe_repr(v)
    return out


def _diff_locals(cur, prev):
    return {k: v for k, v in cur.items() if prev.get(k) != v}


def _fmt_locals(d):
    return ", ".join(f"{k}={v}" for k, v in d.items())


def _indent_of(line):
    return line[: len(line) - len(line.lstrip())]


def _block_keyword(src):
    """Return 'if'|'elif'|'while'|'for' if `src` is a single-line block header
    (ends with ':' and starts with one of those keywords), else None.
    Multi-line conditions (where the first line has no trailing ':') are
    deliberately excluded to keep the heuristic unambiguous."""
    stripped = src.strip()
    if not stripped.endswith(":"):
        return None
    for kw in ("if ", "elif ", "while ", "for "):
        if stripped.startswith(kw):
            return kw.rstrip()
    return None


def _frame_qualname(frame):
    """ClassName.method_name style qualname. Uses co_qualname on 3.11+, falls
    back to inspecting self/cls on older pythons."""
    code = frame.f_code
    qual = getattr(code, "co_qualname", None)
    if qual:
        return qual
    name = code.co_name
    locs = frame.f_locals
    if "self" in locs:
        try:
            return f"{type(locs['self']).__name__}.{name}"
        except Exception:
            pass
    if "cls" in locs:
        try:
            return f"{locs['cls'].__name__}.{name}"
        except Exception:
            pass
    return name


def _is_test_function(qualname, func_name):
    """Pytest convention: test functions and test-class methods start with 'test'."""
    if func_name.startswith("test"):
        return True
    # Method on a Test* class: qualname like "TestFoo.test_bar"
    if "." in qualname:
        head = qualname.split(".", 1)[0]
        if head.startswith("Test") and qualname.rsplit(".", 1)[-1].startswith("test"):
            return True
    return False


def _current_pytest_nodeid():
    current = os.environ.get("PYTEST_CURRENT_TEST")
    if not current:
        return None
    nodeid, _, _phase = current.rpartition(" ")
    return nodeid or current


class _Tracer:
    """Buffers one annotated-source trace per top-level pytest test function.

    Each outermost `test*` call starts a new buffer; all nested calls (helpers,
    fixtures called within) write into that buffer until the test frame returns.
    Events outside any test frame are dropped entirely. The output file is
    JSONL, one row per test: {test_id, file, function, trace, event_count}.
    """

    def __init__(self, output_path, repo_root):
        self.output_path = output_path
        self.repo_root = os.path.realpath(repo_root) if repo_root else None
        self._fp = None
        self._lock = threading.RLock()
        self._stopped = False
        self.event_count = 0
        self.row_count = 0
        # {frame_id: (src, locals_before, func, rel, lineno, iter_count)}
        self._pending = {}
        self._last_emitted_frame = None
        # {frame_id: {lineno: count}} — how many times a line has fired in a
        # given frame; used to emit `iter=N` markers inside loop bodies.
        self._iter_counts = {}
        # {callee_frame_id: caller_locals_snapshot_at_call_time}
        # Used to annotate the caller's post-call locals delta (so the
        # call-site assignment is visible even after we flush the caller
        # line pre-descent). Since the assignment bytecode runs *after* the
        # return event, the annotation gets emitted on the next line event
        # in the caller.
        self._caller_snapshots = {}
        # {caller_frame_id: (indent, pre_call_snapshot)} — set on return,
        # consumed on the next line event in the caller frame.
        self._pending_caller_delta = {}
        self._active_frame_id = None
        self._active_meta = None
        self._buffer = None
        self._test_events_since_shrink = 0
        self._test_max_depth = None  # None = unlimited
        self._test_max_depth_seen = 0
        self._active_thread_ids = set()

    def _install_thread_attribution(self):
        tracer = self
        original_start = threading.Thread.start
        original_run = threading.Thread.run

        def start_with_trace(thread_self, *args, **kwargs):
            with tracer._lock:
                if tracer._active_meta is not None:
                    thread_self._tracer_active_frame_id = tracer._active_frame_id
            return original_start(thread_self, *args, **kwargs)

        def run_with_trace(thread_self, *args, **kwargs):
            frame_id = getattr(thread_self, "_tracer_active_frame_id", None)
            if frame_id is not None:
                with tracer._lock:
                    tracer._active_thread_ids.add(threading.get_ident())
            try:
                return original_run(thread_self, *args, **kwargs)
            finally:
                if frame_id is not None:
                    with tracer._lock:
                        tracer._active_thread_ids.discard(threading.get_ident())

        threading.Thread.start = start_with_trace
        threading.Thread.run = run_with_trace

    def start(self):
        self._fp = open(self.output_path, "w", buffering=1)
        sys.settrace(self._trace)
        threading.settrace(self._trace)
        self._install_thread_attribution()

    def stop(self):
        self._stopped = True
        sys.settrace(None)
        threading.settrace(None)
        if self._active_meta is not None:
            self._finalize_test()
        if self._fp:
            try:
                self._fp.flush()
                self._fp.close()
            finally:
                self._fp = None

    def _depth_from_test(self, frame):
        """Depth of `frame` relative to the active test frame. 0 means we're
        standing in the test function itself; 1 = direct helper; etc."""
        depth = 0
        f = frame
        while f is not None and depth < 200:
            if id(f) == self._active_frame_id:
                return depth
            f = f.f_back
            depth += 1
        if threading.get_ident() in self._active_thread_ids:
            return 1
        return -1  # unreachable — test frame not on stack (shouldn't happen)

    def _trace(self, frame, event, arg):
        with self._lock:
            return self._trace_locked(frame, event, arg)

    def _trace_locked(self, frame, event, arg):
        try:
            if self._stopped or self._fp is None:
                return None
            filename = frame.f_code.co_filename
            if _should_skip(filename, self.repo_root):
                return None

            # Start a new test row when we see an outermost test_* call.
            if (
                event == "call"
                and self._active_meta is None
                and _is_test_function(
                    _frame_qualname(frame),
                    frame.f_code.co_name,
                )
            ):
                self._begin_test(frame, filename)

            if self._active_meta is None:
                return self._trace

            depth = self._depth_from_test(frame)
            if depth < 0:
                return self._trace

            # Honor the active depth cap for everything except the test frame
            # itself returning (we always need that to finalize the row).
            if (
                self._test_max_depth is not None
                and depth > self._test_max_depth
                and not (event == "return" and id(frame) == self._active_frame_id)
            ):
                return self._trace

            if depth > self._test_max_depth_seen:
                self._test_max_depth_seen = depth

            self._emit(frame, event, arg, filename)
            self.event_count += 1
            self._test_events_since_shrink += 1

            if event == "return" and id(frame) == self._active_frame_id:
                self._finalize_test()
            elif self._test_events_since_shrink >= PER_TEST_BUDGET:
                self._shrink_depth()

            return self._trace
        except Exception:
            return self._trace

    def _shrink_depth(self):
        """Pull in the max trace depth by one level. Called when a test has
        burned through PER_TEST_BUDGET events at the current depth cap."""
        cur = self._test_max_depth if self._test_max_depth is not None else self._test_max_depth_seen
        new = max(0, cur - 1)
        self._test_max_depth = new
        self._test_max_depth_seen = new
        self._test_events_since_shrink = 0

    def _begin_test(self, frame, filename):
        rel = os.path.relpath(filename, self.repo_root) if self.repo_root else filename
        qual = _frame_qualname(frame)
        test_id = _current_pytest_nodeid() or f"{rel}::{qual}"
        self._buffer = io.StringIO()
        # Dump the full source of the test function up-front so each row reads
        # as "here's the test, then here's its execution trace".
        try:
            src_lines, _start = inspect.getsourcelines(frame)
            self._buffer.write("# --- test source ---\n")
            for line in src_lines:
                self._buffer.write(line.rstrip() + "\n")
            self._buffer.write("# --- execution trace ---\n")
        except Exception:
            pass
        self._active_meta = {
            "test_id": test_id,
            "file": rel,
            "function": qual,
            "start_events": self.event_count,
        }
        self._active_frame_id = id(frame)
        self._pending.clear()
        self._iter_counts.clear()
        self._caller_snapshots.clear()
        self._pending_caller_delta.clear()
        self._last_emitted_frame = None
        self._test_events_since_shrink = 0
        self._test_max_depth = None
        self._test_max_depth_seen = 0
        self._active_thread_ids = {threading.get_ident()}

    def _finalize_test(self):
        if self._active_frame_id in self._pending:
            src = self._pending.pop(self._active_frame_id)[0]
            self._buffer.write(src + "\n")
        meta = self._active_meta
        trace_text = self._buffer.getvalue()
        row = {
            "test_id": meta["test_id"],
            "file": meta["file"],
            "function": meta["function"],
            "trace": trace_text,
            "event_count": self.event_count - meta["start_events"],
            "final_depth_cap": self._test_max_depth if self._test_max_depth is not None else -1,
        }
        self._fp.write(json.dumps(row) + "\n")
        self.row_count += 1
        self._buffer = None
        self._active_meta = None
        self._active_frame_id = None
        self._pending.clear()
        self._iter_counts.clear()
        self._caller_snapshots.clear()
        self._pending_caller_delta.clear()
        self._last_emitted_frame = None
        self._active_thread_ids.clear()

    def _header(self, rel, func, lineno):
        self._write(f"\n# === {rel}::{func} (line {lineno}) ===\n")

    def _flush_pending(self, frame_id, state_after, next_lineno=None):
        """Emit the buffered line for frame_id using `state_after` to compute
        what locals changed as a result of that line executing. If we know
        the next line about to run in the same frame (only true when flushing
        due to another line event), use that to annotate conditional branches."""
        entry = self._pending.pop(frame_id, None)
        if entry is None:
            return
        src, before, _func, _rel, lineno, iter_count = entry
        parts = []
        delta = _diff_locals(state_after, before)
        if delta:
            parts.append(_fmt_locals(delta))

        if next_lineno is not None:
            kw = _block_keyword(src)
            if kw is not None:
                if next_lineno == lineno + 1:
                    parts.append(f"branch={kw}:True")
                elif next_lineno > lineno + 1:
                    parts.append(f"branch={kw}:False")
                # next_lineno < lineno means loop back — don't annotate.

        if iter_count > 1:
            parts.append(f"iter={iter_count}")

        suffix = "  # " + ", ".join(parts) if parts else ""
        self._write(f"{src}{suffix}\n")
        self._last_emitted_frame = frame_id

    def _emit(self, frame, event, arg, filename):
        rel = os.path.relpath(filename, self.repo_root) if self.repo_root else filename
        lineno = frame.f_lineno
        frame_id = id(frame)
        func = _frame_qualname(frame)
        current = _snapshot_locals(frame)

        if event == "call":
            # Flush the caller's pending line first so the call site anchors
            # the descent visually — otherwise the line that caused the call
            # would render *below* the entire callee trace (off-by-one).
            # Also stash the caller's pre-call locals so we can annotate the
            # RETURN line with the effect of the call on the caller.
            caller_frame = frame.f_back
            if caller_frame is not None:
                caller_snap = _snapshot_locals(caller_frame)
                self._caller_snapshots[frame_id] = caller_snap
                caller_id = id(caller_frame)
                if caller_id in self._pending:
                    self._flush_pending(caller_id, caller_snap)
            src = linecache.getline(filename, lineno).rstrip() or f"# <no source: {rel}:{lineno}>"
            self._header(rel, func, lineno)
            self._write(src + "\n")
            if current:
                indent = _indent_of(src) + "    "
                self._write(f"{indent}# ENTER: {_fmt_locals(current)}\n")
            self._last_emitted_frame = frame_id
            self._iter_counts.pop(frame_id, None)
        elif event == "line":
            # If a call just returned into this frame, emit the caller-locals
            # delta now — the LHS assignment has landed by the time this line
            # event fires.
            delta_entry = self._pending_caller_delta.pop(frame_id, None)
            if delta_entry is not None:
                pre_indent, pre_snap = delta_entry
                delta = _diff_locals(current, pre_snap)
                if delta:
                    self._write(f"{pre_indent}# → {_fmt_locals(delta)}\n")
            # If we're re-entering a different frame (return-to-caller, yield
            # resume), emit a context header BEFORE flushing the pending line
            # so the flushed line appears under its own header.
            crossed_frame = frame_id != self._last_emitted_frame and self._last_emitted_frame is not None
            if crossed_frame:
                self._header(rel, func, lineno)
            # Flush any pending line in this same frame (its effect is now
            # knowable from the current snapshot). We pass the new line number
            # so _flush_pending can annotate conditional branches.
            if frame_id in self._pending:
                self._flush_pending(frame_id, current, next_lineno=lineno)
            src = linecache.getline(filename, lineno).rstrip() or f"# <no source: {rel}:{lineno}>"
            frame_iters = self._iter_counts.setdefault(frame_id, {})
            iter_count = frame_iters.get(lineno, 0) + 1
            frame_iters[lineno] = iter_count
            self._pending[frame_id] = (src, dict(current), func, rel, lineno, iter_count)
            self._last_emitted_frame = frame_id
        elif event == "return":
            if frame_id in self._pending:
                self._flush_pending(frame_id, current)
            rv = _safe_repr(arg)
            frame_src = linecache.getline(filename, frame.f_code.co_firstlineno).rstrip()
            indent = _indent_of(frame_src) + "    " if frame_src else "    "
            self._write(f"{indent}# RETURN from {func}: {rv}\n")
            # Queue caller-delta annotation for the next line event in the
            # caller — at return time the LHS assignment hasn't run yet.
            pre_snap = self._caller_snapshots.pop(frame_id, None)
            if pre_snap is not None and frame.f_back is not None:
                self._pending_caller_delta[id(frame.f_back)] = (indent, pre_snap)
            self._last_emitted_frame = frame_id
            self._iter_counts.pop(frame_id, None)
        elif event == "exception":
            if frame_id in self._pending:
                self._flush_pending(frame_id, current)
            exc_type, exc_value, _ = arg
            name = getattr(exc_type, "__name__", str(exc_type))
            self._write(f"    # EXCEPTION in {func}: {name}: {_safe_repr(exc_value)}\n")

    def _write(self, s):
        with self._lock:
            if self._buffer is not None:
                self._buffer.write(s)


def _neutralize_coverage():
    """pytest-cov / plain `coverage run` both install their own sys.settrace
    hook (via a C extension) which clobbers ours on the exact frames we care
    about — test functions. Neuter coverage.Coverage.start/stop before it
    ever gets a chance to activate. No-op if coverage isn't installed."""
    try:
        import coverage
    except Exception:
        return
    try:
        coverage.Coverage.start = lambda self, *a, **kw: None
        coverage.Coverage.stop = lambda self, *a, **kw: None
        # pytest-cov instantiates coverage via `_start_cov`; the start override
        # above is enough, but also zero out the underlying tracer installers
        # in case anything reaches them directly.
        if hasattr(coverage, "Collector"):
            coverage.Collector.start = lambda self, *a, **kw: None
            coverage.Collector.stop = lambda self, *a, **kw: None
    except Exception:
        pass


def _lock_settrace(tracer):
    """Replace sys.settrace with a no-op so no Python-level caller can
    clobber our trace hook. Covers debuggers, profilers, pytest plugins that
    try to install their own tracer, etc. Does *not* cover C-level callers
    that go through PyEval_SetTrace directly — `_neutralize_coverage` handles
    the one known culprit there."""

    def _locked(_func):
        return

    _locked.__name__ = "settrace"
    sys.settrace = _locked
    real_threading_settrace = threading.settrace

    def _locked_threading(_func):
        real_threading_settrace(tracer._trace)

    # threading.settrace is a separate attribute that threads consult; keep it
    # pinned to our tracer so future threads remain traced.
    threading.settrace = _locked_threading


def _install():
    active_pid = os.environ.get("TRACER_ACTIVE_PID")
    if active_pid is not None and active_pid != str(os.getpid()):
        return
    os.environ["TRACER_ACTIVE_PID"] = str(os.getpid())
    _neutralize_coverage()
    repo_root = os.environ.get("TRACER_REPO_ROOT", os.getcwd())
    output_path = os.environ.get("TRACER_OUTPUT", "/tmp/trace.jsonl")
    tracer = _Tracer(output_path, repo_root)
    tracer.start()
    _lock_settrace(tracer)

    def _on_exit():
        tracer.stop()
        print(
            f"\n[tracer] Wrote {tracer.row_count} test rows " f"({tracer.event_count} events) to {output_path}",
            file=sys.stderr,
        )

    atexit.register(_on_exit)


if __name__ == "sitecustomize":
    _install()
