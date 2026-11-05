# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a Lean proof of a frozen theorem statement on Lean v4.34.1, by kernel replay.

``/tests/config.json`` is the task's contract: the statement file's ``prefix`` (imports, ``open``s and
helper declarations), the theorem ``statement`` through its final ``:=``, and the ``target`` theorem
name. The reply's last ``lean4``/``lean`` code block (else its last code block, else the whole reply)
is either a proof term or, when it declares the target theorem itself, a whole file after the imports.
Both files below are named ``Task.lean``, so that private names agree. Each step runs as its own
unprivileged user, chosen afresh for each grade, after the machine's world-writable directories and
``/tests``, where a source may keep a reference proof hidden from the solver, are closed to those users;
the three steps that write run in a directory only their user can write, and verification writes nothing:

1. The reference user compiles the reference: the statement proved by ``sorry``, then a check, run by
   Lean, that the module declares the target as a theorem, that no declaration's statement uses
   ``sorry``, that only the target's proof does, and that the module declares no axiom
   (``native_decide`` declares one).
2. The candidate user compiles the candidate: the statement followed by the proof term in parentheses, or
   the imports followed by the reply's file. Elaboration runs arbitrary code, including any commands
   the reply contains, so everything this user writes is untrusted data; only the target, as replayed
   below, counts.
3. The export user runs ``lean-export`` from Prime Intellect's checker, which turns the candidate's
   object file into bounded JSON data.
4. The verify user runs ``lean-verify``, which reads only that JSON and the reference object file, and
   replays the target's dependency closure in the kernel against the reference's imports. The target
   scores 1 only if it is accepted in proving mode, with the reference's type and the reference's
   statement declarations unchanged, and its axioms are among ``propext``, ``Classical.choice`` and
   ``Quot.sound``.

Root only orchestrates, and adopts the steps' orphaned processes so that it reaps them whatever the
machine's init does. After each step every process of the step's user is killed and System V IPC
objects are removed, before root copies the step's output, a regular file opened without following
links; while a step runs, its directory's size and file count are checked, a soft bound that a hard
storage limit on the grading machine would make strict. The reward on stdout is 0
only for an outcome the proof caused: the candidate's compile or export failing in any way (the
reference has just compiled on the same machine, and the export reads only the candidate's object file),
the candidate's own time or space allowance exceeded, or a rejection by ``lean-verify``. Any other
failure (a tool or user switch that does not work, a reference that does not validate, ``lean-verify``
failing or exceeding its allowance, which a valid but slow proof can do, a malformed verdict, the overall
deadline) exits non-zero, so the pipeline records an infrastructure error rather than a score. RL
training masks those rollouts; of these outcomes, a reply can reach only ``lean-verify`` running out of
time or failing on the proof data, after its proof has compiled and exported. Details go to stderr as
JSON.
"""

import ctypes
import errno
import json
import os
import re
import resource
import secrets
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

TESTS = Path("/tests")
CONFIG = TESTS / "config.json"
ANSWER = Path("/app/answer.txt")
LEAN_ROOT = Path("/opt/lean")
VERIFY_BIN = LEAN_ROOT / "verify/.lake/build/bin/lean-verify"
EXPORT_BIN = LEAN_ROOT / "verify/.lake/build/bin/lean-export"
SETPRIV = Path("/usr/bin/setpriv")
IPCRM = Path("/usr/bin/ipcrm")
WORK = Path("/tmp/lean-grade")
MODULE_FILE = "Task.lean"
OBJECT_FILE = "Task.olean"
PROOF_DATA = "proof.json"
"""The file ``lean-export`` writes and ``lean-verify`` reads."""
LEAN_VERSION = "4.34.1"
LEAN_THREADS = "--threads=4"
"""Bounds each compile's threads, which count against the step users' process limit."""

FIRST_STEP_UID, STEP_UID_BLOCKS = 20000, 10000
"""Each grade's four step users are a block of consecutive uids from 20000 up, chosen at random: process
limits count every process of a user on the host, so fixed users would let one grade starve another. Two
grades on one host share a block with probability 1 in 10,000, which neither can see."""
PR_SET_CHILD_SUBREAPER = 36
SHARED_DIRECTORIES = ("/tmp", "/var/tmp", "/run/lock", "/dev/shm")
"""The machine's world-writable directories, closed to the step users before any step runs."""
ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
TARGET_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_'.!?]*")
"""A target name that is safe to write into the reference check as a Lean string."""
LEAN_ERROR = re.compile(rb"^" + re.escape(MODULE_FILE.encode()) + rb":\d+:\d+: error", re.MULTILINE)
"""A Lean error diagnostic in the compiled file, which tells a failed compile from a crashed one."""
EXPORT_REJECTIONS = ("malformed object file:", "proof data is too large", "invalid header", "incompatible header")
"""What ``lean-export`` prints when the object file it was given is not one it accepts; any other failure of
the export is a crash on the candidate's object file, which scores 0 as well."""
MISSING_TARGET = "target is not declared in the reference"
"""``lean-verify``'s reason when the reference lacks the target, which the reference check should prevent."""

STEP_SECONDS = {"reference": 90.0, "candidate": 120.0, "export": 30.0, "verify": 90.0}
"""Each step's own allowance. The candidate's compile or export exceeding its allowance scores 0; the
reference or verify step exceeding its allowance is an infrastructure error. The 366 ProofNet-Verified
references compiled in 12.6 s at the median and 33 s at most, so the trusted steps have room for a busy
machine."""
DEADLINE_SECONDS = 360.0
"""From the script's start to its verdict: the step allowances plus the checks and copies between them, so
that it binds only when the machine itself is slow. The task's scorer timeout leaves a margin above it."""
MAX_REPLY = 1 << 20
"""Largest reply read, in bytes."""
MAX_PROOF = 1 << 16
"""Longest proof graded, in bytes."""
MAX_FILE = 1 << 28
"""Largest file a step's user may write."""
MAX_STEP_BYTES = 1 << 29
MAX_STEP_FILES = 2000
"""Bytes and directory entries a step may leave in its directory before it is stopped."""
MAX_OBJECT = 1 << 28
MAX_PROOF_DATA = 1 << 27
"""Largest object file and exported proof data copied between steps, in bytes."""
MAX_PROCESSES = 1024
"""Processes and threads each step's user may run."""
OUTPUT_TAIL = 4000
"""Bytes of a step's output kept for diagnostics."""
FENCE = re.compile(r"```[ \t]*(?P<language>[\w+-]*)[ \t]*\n(?P<code>.*?)```", re.DOTALL)
"""A fenced code block and its language tag."""
IMPORT_LINE = re.compile(r"^[ \t]*import\s.*$", re.MULTILINE)
COMMENT_OR_STRING = re.compile(r'"(?:\\.|\\\Z|[^"\\])*(?:"|\Z)|--[^\n]*|/-', re.DOTALL)
"""A Lean string literal (one left open runs to the end), a line comment, or the start of a block comment.
The patterns that read the reply match within a line or run to the end, so each reads it in linear time."""

REFERENCE_CHECK = """
open Lean in
run_cmd do
  let env ← getEnv
  let target := "{target}".toName
  let some info := env.find? target | throwError "the reference declares no constant {{target}}"
  unless info matches .thmInfo _ do throwError "{{target}} is not a theorem"
  let own := env.constants.map₂.toList
  if let some (name, _) := own.find? (·.2 matches .axiomInfo _) then
    throwError "the reference declares the axiom {{name}}"
  if let some (name, _) := own.find? (·.2.type.hasSorry) then
    throwError "the statement of {{name}} uses sorry"
  let sorried := own.filterMap fun (name, ci) =>
    if ((ci.value? (allowOpaque := true)).map (·.hasSorry)).getD false then some name else none
  unless sorried == [target] do
    throwError "only the proof of {{target}} may be sorry, but these use it: {{sorried}}"
"""
"""Appended to the reference: Lean, not text matching, establishes what the reply must prove."""


class Stop(StrEnum):
    """Why the grader stopped a step before it exited."""

    TIMEOUT = "timeout"
    """The step used its own allowance."""
    DEADLINE = "deadline"
    """The overall deadline came first: an infrastructure error, whichever step it was."""
    QUOTA = "quota"
    """The step's directory grew past its allowance."""


class GraderFailure(Exception):
    """The grader, not the proof, failed: reported as an infrastructure error."""


@dataclass(frozen=True)
class Contract:
    """What the reply must prove: ``statement`` after ``prefix``, with ``target`` the theorem to replay."""

    prefix: str
    statement: str
    target: str

    @classmethod
    def from_config(cls, config: object) -> "Contract":
        if not isinstance(config, dict) or set(config) != {"prefix", "statement", "target"}:
            raise GraderFailure(f"malformed contract: {config!r}"[:500])
        if not all(isinstance(config[key], str) for key in config):
            raise GraderFailure("contract fields must be text")
        contract = cls(**config)
        if TARGET_NAME.fullmatch(contract.target) is None or not contract.statement.rstrip().endswith(":="):
            raise GraderFailure(f"malformed contract target or statement for {contract.target!r}")
        return contract


@dataclass(frozen=True)
class StepResult:
    """How one step ended, and where its output is."""

    exit_code: int | None
    """The step's exit code; ``None`` when the step was stopped."""
    stopped: Stop | None
    """Why the step was stopped; ``None`` when it exited."""
    log: Path
    """The step's stderr, and its stdout unless captured separately."""
    stdout: Path | None

    def tail(self) -> str:
        with self.log.open("rb") as log:
            log.seek(max(0, self.log.stat().st_size - OUTPUT_TAIL))
            return log.read().decode("utf-8", "replace")


def proof_block(reply: str) -> str:
    """The reply's last ``lean4``/``lean`` block, else its last code block, else the whole reply."""
    blocks = [(match["language"].lower(), match["code"]) for match in FENCE.finditer(reply)]
    tagged = [code for language, code in blocks if language in ("lean", "lean4")]
    if tagged:
        return tagged[-1].strip()
    return (blocks[-1][1] if blocks else reply).strip()


def without_comments(source: str) -> str:
    """``source`` with Lean comments, nested block comments included, replaced by spaces; strings are kept."""
    pieces, position = [], 0
    while (match := COMMENT_OR_STRING.search(source, position)) is not None:
        pieces.append(source[position : match.start()])
        token = match.group()
        if token != "/-":
            pieces.append(token if token.startswith('"') else " ")
            position = match.end()
            continue
        depth, index = 1, match.end()
        while depth and index < len(source):
            if source.startswith("/-", index):
                depth, index = depth + 1, index + 2
            elif source.startswith("-/", index):
                depth, index = depth - 1, index + 2
            else:
                index += 1
        pieces.append(" ")
        position = index
    pieces.append(source[position:])
    return "".join(pieces)


def declares_target(block: str, target: str) -> bool:
    """Whether the reply restates the target theorem, outside comments, rather than giving only its proof."""
    declaration = rf"(?<![\w'.!?])(?:theorem|lemma)\s+{re.escape(target)}(?![\w'.!?])"
    return re.search(declaration, without_comments(block)) is not None


def reference_source(contract: Contract) -> str:
    """The statement proved by ``sorry``, followed by the reference check."""
    statement = f"{contract.prefix.rstrip()}\n\n{contract.statement.rstrip()} by\n  sorry\n"
    return statement + REFERENCE_CHECK.format(target=contract.target)


def candidate_source(contract: Contract, block: str) -> str:
    """The statement followed by the proof term in parentheses, or the prefix's imports followed by the
    reply's own file when it restates the target."""
    if declares_target(block, contract.target):
        imports = "\n".join(line.strip() for line in IMPORT_LINE.findall(contract.prefix))
        return f"{imports}\n\n{IMPORT_LINE.sub('', block).strip()}\n"
    body = "\n".join(f"    {line}" if line.strip() else "" for line in block.splitlines())
    return f"{contract.prefix.rstrip()}\n\n{contract.statement.rstrip()}\n  (\n{body}\n  )\n"


def make_step_dir(name: str, uid: int) -> Path:
    """A fresh directory for one step that only ``uid`` can write."""
    path = WORK / name
    path.mkdir(mode=0o700)
    os.chown(path, uid, uid)
    path.chmod(0o700)
    return path


def close_grader_files() -> None:
    """Make ``/tests`` root-only, so that no step reads the task's evidence or the grader's own files."""
    os.chown(TESTS, 0, 0)
    TESTS.chmod(0o700)
    if TESTS.stat().st_mode & 0o077:
        raise GraderFailure(f"{TESTS} is still open to other users")


def close_shared_directories() -> None:
    """Make the machine's world-writable directories writable by root only, so a step writes only its own."""
    for directory in SHARED_DIRECTORIES:
        path = Path(directory)
        if not path.is_dir():
            continue
        path.chmod(0o1755)
        if path.stat().st_mode & stat.S_IWOTH:
            raise GraderFailure(f"{directory} is still world-writable")


def become_subreaper() -> None:
    """Make the grader the parent of every orphaned process a step leaves, so that the grader reaps them."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise GraderFailure(f"cannot adopt the steps' processes: {os.strerror(ctypes.get_errno())}")


def reap_orphans() -> None:
    """Collect every exited process the grader has adopted; called only once the step's own process is collected."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def kill_user(uid: int) -> None:
    """Kill every process of ``uid`` in this PID namespace with one signal sent as that user, and reap them,
    until none is left."""
    for _ in range(50):
        child = os.fork()
        if child == 0:
            try:
                os.setgroups([])
                os.setresgid(uid, uid, uid)
                os.setresuid(uid, uid, uid)
                os.kill(-1, signal.SIGKILL)
            finally:
                os._exit(0)
        os.waitpid(child, 0)
        reap_orphans()
        if not processes_of(uid):
            return
        time.sleep(0.05)
    raise GraderFailure(f"processes of user {uid} survived repeated kills")


def processes_of(uid: int) -> list[int]:
    """The process IDs whose real user is ``uid``; a process that exits while being read is skipped."""
    pids = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            status = Path(entry.path, "status").read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as error:
            if error.errno == errno.ESRCH:
                continue
            raise GraderFailure(f"cannot inspect process {entry.name}: {error}") from error
        uids = next((line.split()[1:] for line in status.splitlines() if line.startswith("Uid:")), [])
        if uids and int(uids[0]) == uid:
            pids.append(int(entry.name))
    return pids


def remove_ipc() -> None:
    """Remove System V IPC objects a step left, which would otherwise outlive it."""
    result = subprocess.run([str(IPCRM), "--all"], capture_output=True, text=True, check=False, timeout=30)
    if result.returncode != 0:
        raise GraderFailure(f"ipcrm failed: {result.stderr.strip()}")


def apply_step_limits() -> None:
    """In each step's child, still as root: cap files, processes and message queues; die first on OOM."""
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE, MAX_FILE))
    resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_MSGQUEUE, (0, 0))
    Path("/proc/self/oom_score_adj").write_text("1000")


def as_user(uid: int) -> list[str]:
    """``setpriv`` options that run the rest of its command as ``uid``, with no groups and no new privileges."""
    return [str(SETPRIV), f"--reuid={uid}", f"--regid={uid}", "--clear-groups", "--no-new-privs"]


def over_quota(directory: Path) -> bool:
    """Whether ``directory`` holds more than the step allowance, counted without following links; a file or
    directory that cannot be measured counts as too much."""
    entries, size, unreadable = 0, 0, []
    for root, directories, files in os.walk(directory, onerror=unreadable.append):
        entries += len(directories) + len(files)
        for name in files:
            try:
                size += os.lstat(os.path.join(root, name)).st_size
            except FileNotFoundError:
                continue
            except OSError:
                return True
        if unreadable or entries > MAX_STEP_FILES or size > MAX_STEP_BYTES:
            return True
    return bool(unreadable)


def has_lean_error(log: Path) -> bool:
    """Whether a compile log reports a Lean error in ``Task.lean``, read in full rather than from its tail."""
    with log.open("rb") as reader:
        carry = b""
        while chunk := reader.read(1 << 20):
            if LEAN_ERROR.search(carry + chunk):
                return True
            carry = chunk[-256:]
    return False


@dataclass(frozen=True)
class StepUsers:
    """The users the four steps run as, for one grade."""

    reference: int
    candidate: int
    export: int
    verify: int

    @classmethod
    def choose(cls) -> "StepUsers":
        first = FIRST_STEP_UID + 4 * secrets.randbelow(STEP_UID_BLOCKS)
        return cls(first, first + 1, first + 2, first + 3)

    def __iter__(self) -> Iterator[int]:
        return iter((self.reference, self.candidate, self.export, self.verify))


class GraderSteps:
    """Runs the grader's steps against one overall deadline."""

    def __init__(self, start: float, lean: str, lean_path: str, users: StepUsers) -> None:
        self.start = start
        self.lean = lean
        self.users = users
        self.environment = {"PATH": f"{Path(lean).parent}:/usr/bin:/bin", "LEAN_PATH": lean_path}
        """``lean-verify`` runs ``lean`` itself, to find the toolchain's library, so the toolchain is on ``PATH``."""
        self.logs = WORK / "logs"
        self.logs.mkdir(mode=0o700)
        self.timings: dict[str, float] = {}

    def compile_argv(self) -> list[str]:
        """The command that compiles ``Task.lean`` in the current directory to ``Task.olean``."""
        return [self.lean, LEAN_THREADS, "-o", OBJECT_FILE, MODULE_FILE]

    def remaining(self) -> float:
        """Seconds left before the overall deadline."""
        return DEADLINE_SECONDS - (time.monotonic() - self.start)

    def run(self, step: str, uid: int, argv: list[str], cwd: Path, capture: bool = False) -> StepResult:
        """Run ``argv`` as ``uid`` in ``cwd``; stop it at its allowance, the overall deadline, or its quota."""
        available = self.remaining()
        if available <= 1:
            raise GraderFailure(f"the {DEADLINE_SECONDS:g} s grading deadline left no time for the {step} step")
        limit, bound = (
            (STEP_SECONDS[step], Stop.TIMEOUT) if STEP_SECONDS[step] < available else (available, Stop.DEADLINE)
        )
        log = self.logs / f"{step}.log"
        stdout = self.logs / f"{step}.out" if capture else None
        started = time.monotonic()
        stopped: Stop | None = None
        with log.open("wb") as err, stdout.open("wb") if stdout else err as out:
            try:
                process = subprocess.Popen(
                    [*as_user(uid), *argv],
                    cwd=cwd,
                    env={**self.environment, "HOME": str(cwd)},
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                    preexec_fn=apply_step_limits,
                )
            except OSError as error:
                raise GraderFailure(f"cannot start the {step} step: {error}") from error
            while process.poll() is None:
                if time.monotonic() - started > limit:
                    stopped = bound
                elif over_quota(cwd):
                    stopped = Stop.QUOTA
                if stopped:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    break
                time.sleep(0.2)
        kill_user(uid)
        remove_ipc()
        self.timings[step] = round(time.monotonic() - started, 2)
        return StepResult(None if stopped else process.returncode, stopped, log, stdout)


def copy_untrusted(source: Path, destination: Path, limit: int) -> bool:
    """Copy a regular file a step wrote, without following links; ``False`` if it is absent, not regular, or too big."""
    try:
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.ELOOP, errno.ENXIO):
            return False
        raise GraderFailure(f"cannot read {source}: {error}") from error
    with os.fdopen(fd, "rb") as reader:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            return False
        data = reader.read(limit + 1)
    if len(data) > limit:
        return False
    destination.write_bytes(data)
    destination.chmod(0o644)
    return True


def check_users(steps: GraderSteps) -> None:
    """Fail as the grader, not as the proof, when a step's user is in use or cannot run the toolchain."""
    for uid in steps.users:
        if processes_of(uid):
            raise GraderFailure(f"user {uid} already runs processes")
        result = subprocess.run(
            [*as_user(uid), steps.lean, "--version"],
            capture_output=True,
            text=True,
            check=False,
            cwd="/",
            env=steps.environment,
            timeout=30,
        )
        if result.returncode != 0 or LEAN_VERSION not in result.stdout:
            raise GraderFailure(f"user {uid} cannot run {steps.lean}: {(result.stdout + result.stderr).strip()[:500]}")


def accepted(verdict: object, target: str) -> bool:
    """Whether ``lean-verify``'s verdict accepts ``target`` as proved with only the standard axioms.

    The verdict must have the shape ``lean-verify`` prints; anything else is the grader's failure, not a
    rejection of the proof.
    """
    if not isinstance(verdict, dict) or not isinstance(verdict.get("ok"), bool):
        raise GraderFailure(f"malformed lean-verify verdict: {verdict!r}"[:2000])
    if verdict.get("reference_error"):
        raise GraderFailure(f"lean-verify rejected the reference: {verdict.get('error')}")
    targets = verdict.get("targets")
    if not isinstance(targets, list) or not all(isinstance(entry, dict) for entry in targets):
        raise GraderFailure(f"malformed lean-verify targets: {targets!r}"[:2000])
    entry = next((entry for entry in targets if entry.get("name") == target), None)
    if entry is None or entry.get("reason") == MISSING_TARGET:
        raise GraderFailure(f"lean-verify found no target {target} in the reference: {verdict.get('error')}")
    axioms, mode = entry.get("axioms"), entry.get("mode")
    if (
        not isinstance(entry.get("ok"), bool)
        or not isinstance(mode, str)
        or not isinstance(axioms, list)
        or not all(isinstance(axiom, str) for axiom in axioms)
    ):
        raise GraderFailure(f"malformed lean-verify verdict for {target}: {entry!r}"[:2000])
    return verdict["ok"] and entry["ok"] and mode == "prove" and set(axioms) <= ALLOWED_AXIOMS


class Rejected(Exception):
    """The proof, not the grader, failed a step: it scores 0, with ``details`` saying why."""

    def __init__(self, **details: object) -> None:
        super().__init__(details.get("status"))
        self.details = details


def compile_reference(contract: Contract, steps: GraderSteps) -> Path:
    """Compile and check the reference; its object file, which only root can read. Fails as the grader."""
    reference = make_step_dir("reference", steps.users.reference)
    (reference / MODULE_FILE).write_text(reference_source(contract))
    (reference / MODULE_FILE).chmod(0o644)
    result = steps.run("reference", steps.users.reference, steps.compile_argv(), reference)
    trusted = WORK / "trusted"
    trusted.mkdir(mode=0o700)
    if result.exit_code != 0 or not copy_untrusted(reference / OBJECT_FILE, trusted / OBJECT_FILE, MAX_OBJECT):
        raise GraderFailure(f"the reference does not validate ({result.stopped or result.exit_code}): {result.tail()}")
    return trusted / OBJECT_FILE


def graded_block(reply: bytes) -> str:
    """The part of the reply that is graded, within the size limits."""
    if len(reply) > MAX_REPLY:
        raise Rejected(status="reply_too_long")
    block = proof_block(reply.decode("utf-8", "replace"))
    if not block:
        raise Rejected(status="empty_proof")
    if len(block.encode()) > MAX_PROOF:
        raise Rejected(status="proof_too_long")
    return block


def compile_candidate(contract: Contract, block: str, steps: GraderSteps, shared: Path) -> Path:
    """Compile the candidate as the candidate user; its object file, copied into ``shared``."""
    candidate = make_step_dir("candidate", steps.users.candidate)
    (candidate / MODULE_FILE).write_text(candidate_source(contract, block))
    (candidate / MODULE_FILE).chmod(0o644)
    result = steps.run("candidate", steps.users.candidate, steps.compile_argv(), candidate)
    if result.stopped == Stop.DEADLINE:
        raise GraderFailure("the grading deadline passed while the candidate compiled")
    if result.stopped:
        raise Rejected(status=f"compile_{result.stopped}", output=result.tail())
    if result.exit_code != 0:
        status = "compile_failed" if result.exit_code == 1 and has_lean_error(result.log) else "compile_crashed"
        raise Rejected(status=status, exit_code=result.exit_code, output=result.tail())
    if not copy_untrusted(candidate / OBJECT_FILE, shared / "candidate.olean", MAX_OBJECT):
        raise Rejected(status="object_missing_or_oversized")
    return shared / "candidate.olean"


def export_candidate(candidate: Path, steps: GraderSteps, shared: Path) -> Path:
    """Export the candidate's object file to proof data as the export user; the data, copied into ``shared``."""
    export = make_step_dir("export", steps.users.export)
    result = steps.run("export", steps.users.export, [str(EXPORT_BIN), str(candidate), str(export / PROOF_DATA)], export)
    if result.stopped == Stop.DEADLINE:
        raise GraderFailure("the grading deadline passed while the candidate was exported")
    if result.stopped:
        raise Rejected(status=f"export_{result.stopped}")
    if result.exit_code != 0:
        output = result.tail()
        rejected = result.exit_code == 1 and any(message in output for message in EXPORT_REJECTIONS)
        raise Rejected(
            status="export_rejected" if rejected else "export_crashed", exit_code=result.exit_code, output=output
        )
    if not copy_untrusted(export / PROOF_DATA, shared / PROOF_DATA, MAX_PROOF_DATA):
        raise Rejected(status="proof_data_missing_or_oversized")
    candidate.unlink()
    return shared / PROOF_DATA


def verify_candidate(contract: Contract, reference: Path, proof_data: Path, steps: GraderSteps) -> tuple[float, dict]:
    """Replay the proof data against the reference as the verify user; the reward and ``lean-verify``'s verdict."""
    shared = proof_data.parent
    (shared / "reference.olean").write_bytes(reference.read_bytes())
    (shared / "reference.olean").chmod(0o644)
    argv = [str(VERIFY_BIN), str(shared / "reference.olean"), str(proof_data), "--target", contract.target]
    result = steps.run("verify", steps.users.verify, argv, shared, capture=True)
    assert result.stdout is not None
    lines = [line for line in result.stdout.read_text(errors="replace").splitlines() if line.startswith("{")]
    if result.exit_code != 0 or not lines:
        raise GraderFailure(f"lean-verify failed ({result.stopped or result.exit_code}): {result.tail()}")
    try:
        verdict = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise GraderFailure(f"lean-verify printed no JSON verdict: {lines[-1][:500]}") from error
    ok = accepted(verdict, contract.target)
    return (1.0 if ok else 0.0), {"status": "accepted" if ok else "rejected", "verdict": verdict}


def grade(contract: Contract, reply: bytes, steps: GraderSteps) -> tuple[float, dict]:
    """The reward and its details: the reference is checked first, so a broken one never yields a score."""
    reference = compile_reference(contract, steps)
    shared = WORK / "shared"
    shared.mkdir(mode=0o755)
    try:
        block = graded_block(reply)
        candidate = compile_candidate(contract, block, steps, shared)
        proof_data = export_candidate(candidate, steps, shared)
    except Rejected as rejection:
        return 0.0, rejection.details
    return verify_candidate(contract, reference, proof_data, steps)


def main() -> None:
    start = time.monotonic()
    contract = Contract.from_config(json.loads(CONFIG.read_text()))
    with ANSWER.open("rb") as answer:
        reply = answer.read(MAX_REPLY + 1)
    try:
        lean = (LEAN_ROOT / "lean-bin").read_text().strip()
        lean_path = (LEAN_ROOT / "lean-path").read_text().strip()
    except OSError as error:
        raise GraderFailure(f"the Lean toolchain is not installed: {error}") from error
    for tool in (Path(lean), VERIFY_BIN, EXPORT_BIN, SETPRIV, IPCRM):
        if not os.access(tool, os.X_OK):
            raise GraderFailure(f"missing grader tool {tool}")
    WORK.mkdir(mode=0o755)
    become_subreaper()
    close_grader_files()
    close_shared_directories()
    resource.setrlimit(resource.RLIMIT_MSGQUEUE, (0, 0))
    steps = GraderSteps(start, lean, lean_path, StepUsers.choose())
    check_users(steps)
    try:
        reward, details = grade(contract, reply, steps)
    finally:
        for uid in steps.users:
            kill_user(uid)
    if steps.remaining() <= 0:
        raise GraderFailure(f"the {DEADLINE_SECONDS:g} s grading deadline passed before the verdict")
    print(json.dumps({**details, "seconds": steps.timings}, ensure_ascii=False, default=str), file=sys.stderr)
    print(reward)


if __name__ == "__main__":
    main()
