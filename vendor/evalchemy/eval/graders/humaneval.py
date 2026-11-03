"""Self-contained grader for the lm-eval-harness ``humaneval`` task.

Ports the candidate construction and ``pass@1`` scoring of
``lm_eval/tasks/humaneval/utils.py`` at the pinned lm-evaluation-harness
``v0.4.12`` so grading needs only the problem text, the model's solution, and
the reference answer. The reference routes through
``evaluate.load("code_eval")``, which downloads a metric module on first use;
this module needs no hub access.

Execution semantics are reproduced exactly, including the sandbox: the candidate
runs in a fresh child process, under ``reliability_guard`` inside a temporary
working directory, with stdio swallowed and a SIGALRM timeout. What is dropped
is pure overhead -- upstream's ``check_correctness`` starts a
``multiprocessing.Manager`` server *in addition to* the worker process, solely
to carry one result string back, so every candidate costs two process spawns
instead of one. A pipe carries the same string.

The per-candidate process is not an optimization target. ``reliability_guard``
mutates its process irreversibly, so a worker cannot be reused, and running
model-generated code in-process would be faster only by removing the isolation
that makes grading safe at all.

The sandbox helpers below are ported from HuggingFace ``evaluate``'s
``code_eval`` metric (Apache-2.0), itself derived from OpenAI's HumanEval
release. As upstream warns, ``reliability_guard`` is a guard against accidental
damage, not a security sandbox; untrusted code still deserves a real one.
"""

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile

TIMEOUT = 3.0
# Upstream waits one second beyond the in-process alarm before killing the
# child, so a wedged candidate is reported as a timeout rather than a hang.
_JOIN_GRACE = 1.0

# A candidate that outruns this is failed deterministically rather than left running.
# 1 GiB fits a real HumanEval solution while bounding a runaway allocator; applied
# best-effort -- the sandbox applies rlimits where the platform allows and degrades
# elsewhere, matching the bounded grader's cap in eval/graders/livecodebench.py.
DEFAULT_MEMORY_BYTES = 1 * 1024 * 1024 * 1024

PASSED = "passed"
TIMED_OUT = "timed out"

# A failure message is candidate-controlled, and the pipe carrying it holds only
# about 64 KiB before send() blocks. The parent is inside join() by then, so a
# larger message would wedge the child until the join expires and leave a partial
# frame behind. Upstream's manager process drains eagerly and does not care;
# truncating keeps a single pipe viable. Only the message is clipped, never the
# verdict, which is decided by comparison against PASSED.
# Leave room for UTF-8 expansion and the pickle frame on small pipe buffers.
_MAX_OUTCOME_CHARS = 512


class TimeoutException(Exception):
    """Raised by :func:`time_limit` when the candidate outlives its SIGALRM window."""


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise OSError

    def readline(self, *args, **kwargs):
        raise OSError

    def readlines(self, *args, **kwargs):
        raise OSError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):  # noqa: N801
    _stream = "stdin"


@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), redirect_stdin(stream):
        yield


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname, chdir(dirname):
        yield dirname


_DISABLED_OS_CALLS = (
    "kill",
    "system",
    "putenv",
    "remove",
    "removedirs",
    "rmdir",
    "fchdir",
    "setuid",
    "fork",
    "forkpty",
    "killpg",
    "rename",
    "renames",
    "truncate",
    "replace",
    "unlink",
    "fchmod",
    "fchown",
    "chmod",
    "chown",
    "chroot",
    "lchflags",
    "lchmod",
    "lchown",
    "getcwd",
    "chdir",
)
_DISABLED_SHUTIL_CALLS = ("rmtree", "move", "chown")


def reliability_guard(maximum_memory_bytes=None):
    """Disable destructive calls so a candidate cannot damage the host.

    Returns a callable that restores the patched `os`/`shutil`/`subprocess` symbols,
    which the caller invokes only after the candidate has finished -- restoring
    earlier would hand the candidate the calls this is meant to withhold.

    Not a security sandbox: it blocks accidents and casual misbehavior, not a
    determined escape. Run untrusted code in a real sandbox as well.
    """
    if maximum_memory_bytes is not None:
        import resource

        # Applied best-effort: Darwin (and some sandboxes) refuse to lower RLIMIT_AS, and
        # the guard must degrade there rather than crash the child -- the SIGALRM and
        # join-grace kill are the bounds that hold everywhere.
        try:
            resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
            if platform.uname().system != "Darwin":
                resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
        except (ValueError, OSError):
            pass

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"

    import shutil
    import subprocess

    saved = []
    for module, names in ((os, _DISABLED_OS_CALLS), (shutil, _DISABLED_SHUTIL_CALLS), (subprocess, ("Popen",))):
        for name in names:
            if hasattr(module, name):
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, None)

    builtins.help = None

    import sys

    for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[name] = None

    def restore():
        for module, name, original in saved:
            setattr(module, name, original)

    return restore


def _run_candidate(check_program, connection, timeout, maximum_memory_bytes):
    """Execute one candidate in this (throwaway) process and report the outcome."""
    with create_tempdir():
        restore = reliability_guard(maximum_memory_bytes)

        try:
            exec_globals = {}
            with swallow_io(), time_limit(timeout):
                exec(check_program, exec_globals)  # noqa: S102  # the graded candidate
            outcome = PASSED
        except TimeoutException:
            outcome = TIMED_OUT
        except BaseException as exc:  # noqa: BLE001  # any failure is a failed candidate
            outcome = f"failed: {exc}"[:_MAX_OUTCOME_CHARS]

        restore()
        # Reported from inside the sandbox, where upstream appends to its manager
        # list. A candidate that leaves a file behind can make tempdir cleanup
        # raise, and the verdict has to be on the wire before that can happen.
        connection.send(outcome)
        connection.close()


def check_correctness(
    check_program: str,
    timeout: float = TIMEOUT,
    maximum_memory_bytes: int = DEFAULT_MEMORY_BYTES,
) -> str:
    """Run one candidate program in an isolated process and return its outcome.

    ``maximum_memory_bytes`` is wired by default so a candidate that outruns it is
    failed deterministically rather than left running unbounded; a caller who never passes
    one still gets the cap.
    """
    receiver, sender = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_run_candidate,
        args=(check_program, sender, timeout, maximum_memory_bytes),
    )
    process.start()
    sender.close()
    process.join(timeout=timeout + _JOIN_GRACE)
    if process.is_alive():
        process.kill()
        process.join()

    outcome = TIMED_OUT
    if receiver.poll():
        try:
            outcome = receiver.recv()
        except (EOFError, OSError):
            # A child killed mid-write leaves a partial frame; score it a
            # timeout rather than propagating into the caller's grade.
            outcome = TIMED_OUT
    receiver.close()
    return outcome


def build_candidate(problem: str, solution: str) -> str:
    """Join the prompt and the completion, as the task's ``create_test`` filter does."""
    return problem + solution


def grade(problem: str, solution: str, reference_answer: str, timeout: float = TIMEOUT) -> float:
    """Grade one completion by running it against the reference tests.

    Args:
        problem: The HumanEval prompt, which the completion continues.
        solution: The model's completion.
        reference_answer: The task's target, ``"{test}\\ncheck({entry_point})"``.
        timeout: Seconds the candidate may run before it is scored a timeout.

    Returns:
        ``1.0`` when the candidate passes the tests, else ``0.0``. This is
        ``pass@1`` for the single sample the task generates.
    """
    check_program = build_candidate(problem, solution) + "\n" + reference_answer
    return 1.0 if check_correctness(check_program, timeout) == PASSED else 0.0
