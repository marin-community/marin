"""Self-contained, bounded grader for the LiveCodeBench (LCB) test format.

One grader replaces the four vendored LCB execution stacks (``LiveCodeBench``,
``LiveCodeBenchv5``, ``LiveCodeBenchv5_official``, ``LiveBench/lcb_runner``). Each candidate
runs in one child process with bounded memory and wall-clock time, and the parent reaps it
deterministically, so a runaway candidate is bounded wherever the RL reward path (or a
benchmark) grades it.

The shape follows ``eval.graders.humaneval`` -- one child process, pipe transport, watchdog
kill, deterministic reap -- with the bounds the vendored stacks lacked:

- a memory cap wired by default (``RLIMIT_AS``/``RLIMIT_DATA``/``RLIMIT_STACK``), applied
  best-effort (some platforms, e.g. Darwin, refuse to lower it, and the grader must not crash
  on that guard there);
- a wall-clock deadline *independent of the test count*, in addition to the per-test
  ``SIGALRM``;
- deterministic child reaping with no Manager server to leak;
- child identification at spawn (pid, test count, deadline).

It is library-consumable, not only route-internal: an RL reward path can call :func:`run`
synchronously per trajectory and derive a binary or fractional reward from the per-test
outcomes. Inputs are the normalized LCB test list -- call-based or stdin, with an optional
``fn_name`` -- plus the extracted program; both stop-on-failure short-circuiting and
collect-all are preserved. Concurrent callers are supported: each run is its own process and
the parent never waits unbounded.
"""

import builtins
import contextlib
import faulthandler
import io
import logging
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types

from eval.runtime.bounded_process import bounded_process
from eval.runtime.resources import ExecutionMode

try:
    import resource  # POSIX-only; the memory cap degrades where it is absent (see apply_memory_cap)
except ImportError:  # pragma: no cover - non-POSIX platforms
    resource = None

logger = logging.getLogger(__name__)

# A candidate that outruns this is failed deterministically. 1 GiB fits a real LCB solution
# and its test harness; it bounds a runaway allocator without squeezing a legitimate one.
DEFAULT_MAX_MEMORY_BYTES = 1 * 1024 * 1024 * 1024

# Per-test alarm. The wall-clock deadline (``DEFAULT_DEADLINE``) bounds the whole run
# independently of how many tests there are.
TIMEOUT = 6.0
DEFAULT_DEADLINE = 30.0

# The parent gives a killed child this much to die and be reaped before giving up; a
# SIGKILL'd process reaps immediately, so this only matters for the (rare) failed kill.
_JOIN_GRACE = 1.0

# Bound candidate-controlled diagnostics while the parent drains test events.
_MAX_DETAIL_CHARS = 8192

PASSED = "passed"
FAILED = "failed"
TIMED_OUT = "timeout"


class TimeoutException(Exception):
    """Raised by :func:`_time_limit` when a single test outlives its SIGALRM window."""


class redirect_stdin(contextlib._RedirectStream):  # noqa: N801
    """``contextlib`` ships ``redirect_stdout``/``redirect_stderr`` only; this is the stdin twin."""

    _stream = "stdin"


def apply_memory_cap(maximum_memory_bytes) -> bool:
    """Apply the child memory cap best-effort.

    Returns True when every rlimit was set. Darwin and some sandboxes refuse to lower
    ``RLIMIT_AS`` and the cap does not bind -- the wall-clock deadline + kill remains the
    bound that holds everywhere, so this degrades instead of crashing the child.
    """
    if not maximum_memory_bytes:
        return False
    if resource is None:  # pragma: no cover - non-POSIX platforms
        return False
    try:
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
        return True
    except (ValueError, OSError):
        return False


def terminate(pid) -> None:
    """SIGKILL a grader child; a no-op if it is already gone."""
    if pid is None:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


@contextlib.contextmanager
def _time_limit(seconds):
    """Bound one test with a SIGALRM (main thread of the child process only)."""

    def _handler(signum, frame):
        raise TimeoutException()

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _reliability_guard(maximum_memory_bytes):
    """Disable destructive calls inside a candidate and (best-effort) cap its memory.

    Returns a restore callable; the caller invokes it only after the candidate has
    finished -- restoring earlier would hand back the calls this is meant to withhold.
    Not a security sandbox: it blocks accidents and casual misbehavior, not a determined
    escape.
    """
    apply_memory_cap(maximum_memory_bytes)
    faulthandler.disable()

    builtins.exit = None
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"

    saved = []
    _DISABLED = {
        os: ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork",
             "forkpty", "killpg", "rename", "renames", "truncate", "replace", "unlink", "chdir",
             "getcwd", "chroot"),
        shutil: ("rmtree", "move", "chown"),
        subprocess: ("Popen",),
    }
    for module, names in _DISABLED.items():
        for name in names:
            if hasattr(module, name):
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, None)
    for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[name] = None

    def restore():
        for module, name, original in saved:
            setattr(module, name, original)

    return restore


def _resolve_call_target(ns, fn_name):
    """Find the candidate object to call, by name or by inferring the first defined function."""
    if fn_name:
        return ns[fn_name]
    for name, value in ns.items():
        if not name.startswith("_") and isinstance(value, (types.FunctionType, types.MethodType)):
            return value
    target = next((v for n, v in ns.items() if isinstance(v, type)), None)
    if target is not None:
        return target()
    raise RuntimeError("no call target inferred from candidate")


def _score(value, expected, comparison="coerced"):
    """Compare a candidate output to its expected value, tolerating int/str/float and list/tuple."""
    if comparison == "exact":
        return value == expected
    if comparison == "stdout_normalized":
        return _normalize_stdout(value) == _normalize_stdout(expected)
    if value == expected:
        return True
    try:
        if isinstance(expected, (list, tuple)) and isinstance(value, (list, tuple)):
            if [int(x) for x in expected] == [int(y) for y in value]:
                return True
        if float(value) == float(expected):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _normalize_stdout(value):
    """Match CodeForces' published stdout comparison before exact equality."""
    return str(value).strip().replace("\n", " ").replace("\r", "")


def _run_stdin(completion, test_input):
    """Run a stdin candidate and return its captured stdout."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(io.StringIO()), redirect_stdin(io.StringIO(test_input)):
        exec(completion, {})  # noqa: S102  # the graded candidate, run per test
    return captured.getvalue().strip()


def _run_call(target, test_input):
    """Call a call-based candidate with its arguments and return the result."""
    if isinstance(test_input, (list, tuple)):
        return target(*test_input)
    if isinstance(test_input, dict):
        return target(**test_input)
    return target(test_input)


def _run_tests(tests, completion, fn_name, stdin_mode, test_timeout, collect_all, connection, cap, comparison):
    """Child side: run the candidate's tests one bounded test at a time over ``connection``."""
    scratch = tempfile.mkdtemp(prefix="lcb-grader-")
    os.chdir(scratch)
    restore = _reliability_guard(cap)
    try:
        target = None
        if not stdin_mode:
            ns = {}
            exec(completion, ns)  # noqa: S102  # define the candidate's functions once
            target = _resolve_call_target(ns, fn_name)
        for index, test in enumerate(tests):
            input_value = test.get("input")
            expected = test.get("output")
            status = FAILED
            detail = ""
            with _time_limit(test_timeout):
                try:
                    if stdin_mode:
                        output = _run_stdin(completion, input_value)
                        status = PASSED if _score(output, expected, comparison) else FAILED
                    else:
                        output = _run_call(target, input_value)
                        status = PASSED if _score(output, expected, comparison) else FAILED
                except TimeoutException:
                    status = TIMED_OUT
                    detail = "per-test alarm expired"
                except BaseException as exc:  # noqa: BLE001  # any candidate failure is a failed test
                    status = FAILED
                    detail = f"{type(exc).__name__}: {exc}"
            connection.send(("test", index, status, detail[:_MAX_DETAIL_CHARS]))
            if status != PASSED and not collect_all:
                break
    finally:
        restore()
        connection.send(("done",))  # the parent treats a missing/failed done as a timeout
        connection.close()


def run(
    tests,
    completion,
    fn_name=None,
    stdin_mode=False,
    test_timeout=TIMEOUT,
    deadline=DEFAULT_DEADLINE,
    collect_all=False,
    maximum_memory_bytes=DEFAULT_MAX_MEMORY_BYTES,
    comparison="coerced",
) -> dict:
    """Run a candidate program against a normalized LCB test list, bounded.

    Args:
        tests: The normalized LCB tests, ``[{"input": ..., "output": ...}, ...]``.
        completion: The extracted program (the candidate's functions or, for stdin, its script).
        fn_name: The function to call; inferred from the candidate when omitted.
        stdin_mode: Feed ``input`` to stdin and compare stdout rather than calling a function.
        test_timeout: Seconds per test before a single test is scored ``timeout``.
        deadline: Wall-clock bound for the whole run, independent of the test count.
        collect_all: On a failure, run the remaining tests; otherwise stop at the first.
        maximum_memory_bytes: The memory cap applied to the child; best-effort per platform.
        comparison: ``exact`` preserves strict-output benchmarks; ``coerced`` uses the LCB comparison.

    Returns:
        A dict with ``passed``, ``num_passed``, ``num_tests``, ``timed_out``,
        ``child_pid``, and per-test ``outcomes`` (``{"index", "outcome", "detail"}``) for a
        binary/fractional reward; ``timed_out`` is True when any test hit the alarm or the run
        was cut off by the deadline.
    """
    tests = list(tests)
    if deadline is None:
        deadline = DEFAULT_DEADLINE

    receiver, sender = multiprocessing.Pipe(duplex=False)
    process = bounded_process(
        target=_run_tests,
        args=(
            tests,
            completion,
            fn_name,
            stdin_mode,
            test_timeout,
            collect_all,
            sender,
            maximum_memory_bytes,
            comparison,
        ),
    )
    process.start()
    sender.close()
    pid = process.pid
    # Child identification at spawn -- the pid, test count, and deadline are what the
    # reward path prints when a runaway trips the watchdog. A test captures the log to
    # check the fields appear.
    logger.debug(
        "lcb grader spawned pid=%s tests=%d test_timeout=%.2fs deadline=%.2fs cap=0x%x",
        pid,
        len(tests),
        test_timeout,
        deadline,
        maximum_memory_bytes,
    )

    watchdog = threading.Timer(deadline, terminate, args=(pid,))
    watchdog.daemon = True
    watchdog.start()

    outcomes: list[dict] = []
    done = False
    deadline_end = time.monotonic() + deadline + _JOIN_GRACE
    try:
        # A child may produce more events than fit in the pipe. Drain while it
        # runs so reporting successful tests cannot itself trigger the watchdog.
        while time.monotonic() < deadline_end:
            if receiver.poll(0.02):
                try:
                    message = receiver.recv()
                except (EOFError, OSError, ConnectionResetError):
                    break  # child died mid-write: partial frame; bounded, no hang
                if message[0] == "test":
                    _, index, status, detail = message
                    outcomes.append({"index": index, "outcome": status, "detail": detail})
                elif message[0] == "done":
                    done = True
                    break
            elif not process.is_alive():
                break
    finally:
        watchdog.cancel()
        if not done and process.is_alive():
            terminate(pid)
        process.join(timeout=_JOIN_GRACE)
        if process.is_alive():
            terminate(pid)
            process.join(timeout=_JOIN_GRACE)
        receiver.close()

    timed_out = (not done) or any(o["outcome"] == TIMED_OUT for o in outcomes)
    num_passed = sum(1 for o in outcomes if o["outcome"] == PASSED)
    all_passed = (
        done
        and len(outcomes) == len(tests)
        and {o["index"] for o in outcomes} == set(range(len(tests)))
        and all(o["outcome"] == PASSED for o in outcomes)
    )

    return {
        "passed": all_passed,
        "num_passed": num_passed,
        "num_tests": len(tests),
        "timed_out": timed_out,
        "child_pid": pid,
        "outcomes": outcomes,
    }


def _infer_call_target_from_completion(completion):
    """Guess the first user-defined function name in a completion (best-effort).

    Mirrors the logic the vendored stacks used before this route existed.
    """
    prefix = completion.split("(")[0].split()
    return prefix[-1] if prefix else None


def run_lcb_tests(problem, completion, timeout):
    """Score a normalized LCB problem through the bounded grader.

    Arguments:
        problem: ``{"test": [{"input":..., "output":..., "testtype":"functional"|"stdin"}...]}``
            (or ``problem["test"]`` as a JSON string, as produced by the dataset loaders).
        completion: The candidate's program (functions or a stdin script).
        timeout: Per-test wall-clock bound.

    Returns per-test ``(passed, message, value, elapsed)`` tuples.
    """
    tests = problem["test"]
    if isinstance(tests, str):
        import json

        tests = json.loads(tests)
    if not tests:
        return []
    is_stdin = tests[0].get("testtype") != "functional"
    normalized = [{"input": tc.get("input"), "output": tc.get("output")} for tc in tests]
    fn_name = None if is_stdin else _infer_call_target_from_completion(completion)
    if not is_stdin and fn_name is None:
        return [(False, "No callable name in completion", "", 0.0) for _ in tests]
    result = run(
        normalized,
        completion,
        fn_name=fn_name,
        stdin_mode=is_stdin,
        test_timeout=timeout,
        deadline=max(timeout * 2, DEFAULT_DEADLINE),
    )
    per_test = result["outcomes"]
    rows = []
    for i, _tc in enumerate(tests):
        if i < len(per_test):
            row = per_test[i]
            rows.append((row["outcome"] == PASSED, row.get("detail", ""), row.get("detail", ""), 0.0))
        else:
            rows.append((False, "Time out!.", "Error: Time out!", float("inf")))
    return rows


def grade(tests, completion, **kwargs) -> float:
    """Return the fraction of tests the candidate passed, for a binary/fractional reward."""
    result = run(tests, completion, **kwargs)
    if result["num_tests"] == 0:
        return 0.0
    return result["num_passed"] / result["num_tests"]


# The shared grading runtime registers program graders with process isolation in
# eval.graders.specs. Direct callers of run also execute candidates in owned processes.
GRADER_EXECUTION_MODE = ExecutionMode.PROCESS_ISOLATED
