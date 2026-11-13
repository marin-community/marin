# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a Lean proof of a frozen theorem statement by kernel replay.

The task's contract, ``/tests/config.json``, holds the statement file's ``prefix`` (imports, ``open``s and
helper declarations), the theorem ``statement`` up to its final ``:=``, and the ``target`` theorem's name.
The graded part of the reply is its last ``lean4`` or ``lean`` code block, else its last code block, else
the whole reply. It is either a proof term or, if it declares the target itself, a whole file after the
imports.

Each grade runs in a fresh sandbox, in four steps:

1. Root compiles the reference: the statement proved by ``sorry``, plus a check, run by Lean, that the
   target is a theorem, that only its proof uses ``sorry``, and that the file declares no axiom
   (``native_decide`` adds one).
2. A random unprivileged uid compiles the candidate. Compiling runs any code the reply contains, so
   everything this uid writes is untrusted. Both files are named ``Task.lean``, so private names agree.
3. The same uid runs Prime Intellect's ``lean-export``, which turns the compiled file into JSON proof data,
   and root copies the data.
4. Root runs ``lean-verify``, which replays the target in the kernel against the reference. The proof
   scores 1 only if it proves the reference's statement, keeps the reference's other declarations, and
   uses no axioms beyond ``propext``, ``Classical.choice`` and ``Quot.sound``.

Before step 1, ``/tests``, which can hold a reference proof the reply must not see, is made root-only, and
the machine's world-writable directories are closed to the candidate's uid. The sandbox itself stops every
process and frees the disk when the grade ends.

The reward on stdout is 0 for anything the proof causes: a failed or crashed compile or export, running
out of time or space, missing proof data, or a rejection by ``lean-verify``. Any other failure (a broken
reference or tool, ``lean-verify`` failing or timing out, a malformed verdict, the overall deadline) exits
non-zero, so the pipeline records an infrastructure error. RL training skips those rollouts, and a reply
can reach one only with proof data that ``lean-verify`` cannot finish checking. Details go to stderr as
JSON.
"""

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
WORK = Path("/tmp/lean-grade")
MODULE_FILE = "Task.lean"
OBJECT_FILE = "Task.olean"
PROOF_DATA = "proof.json"
"""The file ``lean-export`` writes and ``lean-verify`` reads."""
LEAN_VERSION = "4.34.1"
"""The Lean release ``lean.Dockerfile`` installs (``LEAN_TOOLCHAIN``); the candidate's uid must run it."""
LEAN_THREADS = "--threads=4"
"""Bounds each compile's threads, which count against the candidate uid's process limit."""

CANDIDATE_UIDS = range(20000, 60000)
"""Where each grade's candidate uid is drawn from at random: process limits count every process of a uid
on the host, so a fixed uid would let one grade starve another. Two grades on one host share a uid with
probability 1 in 40,000, which neither can see."""
SHARED_DIRECTORIES = ("/tmp", "/var/tmp", "/run/lock", "/dev/shm")
"""The machine's world-writable directories, closed to the candidate's uid before any step runs."""
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
reference or verify step exceeding its allowance is an infrastructure error. The ProofNet-Verified
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
"""Largest file the candidate's uid may write."""
MAX_PROOF_DATA = 1 << 27
"""Largest exported proof data root copies for ``lean-verify``, in bytes."""
MAX_PROCESSES = 1024
"""Processes and threads the candidate's uid may run."""
OUTPUT_TAIL = 4000
"""Bytes of a step's output kept for diagnostics."""
FENCE = re.compile(r"```[ \t]*+(?P<language>[\w+-]*+)[ \t]*+\n(?P<code>.*?)```", re.DOTALL)
"""A fenced code block and its language tag; the possessive quantifiers keep a long run of blanks linear."""
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


def without_comments(source: str, keep_strings: bool = True) -> str:
    """``source`` with Lean comments, nested block comments included, replaced by spaces; strings are kept,
    or emptied to ``""`` unless ``keep_strings``."""
    pieces, position = [], 0
    while (match := COMMENT_OR_STRING.search(source, position)) is not None:
        pieces.append(source[position : match.start()])
        token = match.group()
        if token != "/-":
            string = token if keep_strings else '""'
            pieces.append(string if token.startswith('"') else " ")
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
    """Whether the reply restates the target theorem, outside comments and strings, rather than giving only its
    proof."""
    declaration = rf"(?<![\w'.!?])(?:theorem|lemma)\s+{re.escape(target)}(?![\w'.!?])"
    return re.search(declaration, without_comments(block, keep_strings=False)) is not None


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


def make_step_dir(name: str, uid: int = 0) -> Path:
    """A fresh directory that only ``uid`` can write."""
    path = WORK / name
    path.mkdir(mode=0o700)
    os.chown(path, uid, uid)
    path.chmod(0o700)
    return path


def close_grader_files() -> None:
    """Make ``/tests`` root-only: the candidate's compile runs in this sandbox, next to the task's evidence."""
    os.chown(TESTS, 0, 0)
    TESTS.chmod(0o700)
    if TESTS.stat().st_mode & 0o077:
        raise GraderFailure(f"{TESTS} is still open to other users")


def close_shared_directories() -> None:
    """Make the machine's world-writable directories writable by root only, so the candidate writes only its own."""
    for directory in SHARED_DIRECTORIES:
        path = Path(directory)
        if not path.is_dir():
            continue
        path.chmod(0o1755)
        if path.stat().st_mode & stat.S_IWOTH:
            raise GraderFailure(f"{directory} is still world-writable")


# TODO: the candidate's own uid is what separates its compile and export from verification inside one
# sandbox. Running those steps in their own environment, once task curation can run a post-response step
# there, would isolate the two halves of grading properly and replace the uid switch.
def apply_candidate_limits() -> None:
    """In each of the candidate's steps, still as root: cap files and processes; die first on OOM."""
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE, MAX_FILE))
    resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    Path("/proc/self/oom_score_adj").write_text("1000")


def as_user(uid: int) -> list[str]:
    """``setpriv`` options that run the rest of its command as ``uid``, with no groups and no new privileges."""
    return [str(SETPRIV), f"--reuid={uid}", f"--regid={uid}", "--clear-groups", "--no-new-privs"]


def has_lean_error(log: Path) -> bool:
    """Whether a compile log reports a Lean error in ``Task.lean``, read in full rather than from its tail."""
    with log.open("rb") as reader:
        carry = b""
        while chunk := reader.read(1 << 20):
            if LEAN_ERROR.search(carry + chunk):
                return True
            carry = chunk[-256:]
    return False


class GraderSteps:
    """Runs the grader's steps against one overall deadline."""

    def __init__(self, start: float, lean: str, lean_path: str, uid: int) -> None:
        self.start = start
        self.lean = lean
        self.uid = uid
        """The candidate's uid."""
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

    def run(
        self, step: str, argv: list[str], cwd: Path, *, candidate: bool = False, capture: bool = False
    ) -> StepResult:
        """Run ``argv`` in ``cwd``, as the candidate's uid within its limits or as root; stop it at its
        allowance or the overall deadline."""
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
                    [*as_user(self.uid), *argv] if candidate else argv,
                    cwd=cwd,
                    env={**self.environment, "HOME": str(cwd)},
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                    preexec_fn=apply_candidate_limits if candidate else None,
                )
            except OSError as error:
                raise GraderFailure(f"cannot start the {step} step: {error}") from error
            while process.poll() is None:
                if time.monotonic() - started > limit:
                    stopped = bound
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    break
                time.sleep(0.2)
        self.timings[step] = round(time.monotonic() - started, 2)
        return StepResult(None if stopped else process.returncode, stopped, log, stdout)


def copy_untrusted(source: Path, destination: Path, limit: int) -> bool:
    """Copy a regular file the candidate wrote, without following links; ``False`` if it is absent, not regular,
    too big, or cannot be read or written, which only the candidate's steps can cause."""
    try:
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as reader:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                return False
            data = reader.read(limit + 1)
        if len(data) > limit:
            return False
        destination.write_bytes(data)
    except OSError:
        return False
    return True


def check_candidate_uid(steps: GraderSteps) -> None:
    """Fail as the grader, not as the proof, when the candidate's uid cannot run the toolchain."""
    result = subprocess.run(
        [*as_user(steps.uid), steps.lean, "--version"],
        capture_output=True,
        text=True,
        check=False,
        cwd="/",
        env=steps.environment,
        timeout=30,
    )
    if result.returncode != 0 or LEAN_VERSION not in result.stdout:
        raise GraderFailure(f"uid {steps.uid} cannot run {steps.lean}: {(result.stdout + result.stderr).strip()[:500]}")


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
    """Compile and check the reference as root; its object file, which only root can read. Fails as the grader."""
    reference = make_step_dir("reference")
    (reference / MODULE_FILE).write_text(reference_source(contract))
    result = steps.run("reference", steps.compile_argv(), reference)
    if result.exit_code != 0 or not (reference / OBJECT_FILE).is_file():
        raise GraderFailure(f"the reference does not validate ({result.stopped or result.exit_code}): {result.tail()}")
    return reference / OBJECT_FILE


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


def build_candidate(contract: Contract, block: str, steps: GraderSteps) -> Path:
    """Compile the candidate and export its proof data as the candidate's uid; the data, copied where only
    root can write."""
    candidate = make_step_dir("candidate", steps.uid)
    (candidate / MODULE_FILE).write_text(candidate_source(contract, block))
    (candidate / MODULE_FILE).chmod(0o644)
    compiled = steps.run("candidate", steps.compile_argv(), candidate, candidate=True)
    if compiled.stopped == Stop.DEADLINE:
        raise GraderFailure("the grading deadline passed while the candidate compiled")
    if compiled.stopped:
        raise Rejected(status=f"compile_{compiled.stopped}", output=compiled.tail())
    if compiled.exit_code != 0:
        failed = compiled.exit_code == 1 and has_lean_error(compiled.log)
        status = "compile_failed" if failed else "compile_crashed"
        raise Rejected(status=status, exit_code=compiled.exit_code, output=compiled.tail())
    argv = [str(EXPORT_BIN), OBJECT_FILE, PROOF_DATA]
    exported = steps.run("export", argv, candidate, candidate=True)
    if exported.stopped == Stop.DEADLINE:
        raise GraderFailure("the grading deadline passed while the candidate was exported")
    if exported.stopped:
        raise Rejected(status=f"export_{exported.stopped}")
    if exported.exit_code != 0:
        output = exported.tail()
        rejected = exported.exit_code == 1 and any(message in output for message in EXPORT_REJECTIONS)
        status = "export_rejected" if rejected else "export_crashed"
        raise Rejected(status=status, exit_code=exported.exit_code, output=output)
    verify = make_step_dir("verify")
    if not copy_untrusted(candidate / PROOF_DATA, verify / PROOF_DATA, MAX_PROOF_DATA):
        raise Rejected(status="proof_data_missing_or_unreadable")
    return verify / PROOF_DATA


def verdict_line(output: str) -> str | None:
    """The last line of ``lean-verify``'s output that starts a JSON object. Lines end only at ``\\n``, since a
    name in the verdict may hold another line separator."""
    lines = [line for line in output.split("\n") if line.startswith("{")]
    return lines[-1] if lines else None


def verify_candidate(contract: Contract, reference: Path, proof_data: Path, steps: GraderSteps) -> tuple[float, dict]:
    """Replay the proof data against the reference as root; the reward and ``lean-verify``'s verdict."""
    argv = [str(VERIFY_BIN), str(reference), str(proof_data), "--target", contract.target]
    result = steps.run("verify", argv, proof_data.parent, capture=True)
    assert result.stdout is not None
    line = verdict_line(result.stdout.read_text(errors="replace"))
    if result.exit_code != 0 or line is None:
        raise GraderFailure(f"lean-verify failed ({result.stopped or result.exit_code}): {result.tail()}")
    try:
        verdict = json.loads(line)
    except json.JSONDecodeError as error:
        raise GraderFailure(f"lean-verify printed no JSON verdict: {line[:500]}") from error
    ok = accepted(verdict, contract.target)
    return (1.0 if ok else 0.0), {"status": "accepted" if ok else "rejected", "verdict": verdict}


def grade(contract: Contract, reply: bytes, steps: GraderSteps) -> tuple[float, dict]:
    """The reward and its details: the reference is checked first, so a broken one never yields a score."""
    reference = compile_reference(contract, steps)
    try:
        proof_data = build_candidate(contract, graded_block(reply), steps)
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
    for tool in (Path(lean), VERIFY_BIN, EXPORT_BIN, SETPRIV):
        if not os.access(tool, os.X_OK):
            raise GraderFailure(f"missing grader tool {tool}")
    WORK.mkdir(mode=0o755)
    close_grader_files()
    close_shared_directories()
    steps = GraderSteps(start, lean, lean_path, CANDIDATE_UIDS[secrets.randbelow(len(CANDIDATE_UIDS))])
    check_candidate_uid(steps)
    reward, details = grade(contract, reply, steps)
    if steps.remaining() <= 0:
        raise GraderFailure(f"the {DEADLINE_SECONDS:g} s grading deadline passed before the verdict")
    print(json.dumps({**details, "seconds": steps.timings}, ensure_ascii=False, default=str), file=sys.stderr)
    print(reward)


if __name__ == "__main__":
    main()
