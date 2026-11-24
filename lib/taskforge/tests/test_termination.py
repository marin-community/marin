# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SIGTERM is a crash: the driver exits 143 at once, every ``run`` child follows its parent however the parent died,
and a relaunch reopens each root's last committed head. The process tests signal real driver processes
(``termination_probe``); a probe's stdout reaches end of file only once it and every child are gone."""

import asyncio
import contextlib
import fcntl
import json
import os
import queue
import resource
import select
import signal
import subprocess
import sys
import threading
import time
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
from lib.taskforge.tests.resource_doubles import FakeInference
from lib.taskforge.tests.termination_probe import MODULE
from lib.taskforge.tests.test_end_to_end import SETTINGS, Envs, published, published_workpiece, resources, script

from taskforge import driver
from taskforge.config import TaskforgeConfig
from taskforge.effects.issues import Hold
from taskforge.effects.log import Status
from taskforge.llm.glm_provider import EndpointSource
from taskforge.model.proposal import TASK_PROPOSAL, TaskProposal
from taskforge.model.review import Incomplete
from taskforge.model.workpiece import Terminal
from taskforge.policy import Policy
from taskforge.resources.bundle import Resources
from taskforge.wiring import build_config

ROOT = Path(__file__).parents[3]
STARTUP = 120.0
"""Seconds a probe process may take to import and reach the point it announces, or to run to its end."""
PROMPT = 5.0
"""Seconds within which a signalled driver has exited."""
FOLLOW = 30.0
"""Seconds within which every child has followed its dead parent (a child may still be importing)."""
TERMINATED = 143


def inputs(proposal: TaskProposal, directory: Path, names: tuple[str, ...] = ("test",)) -> list[Path]:
    (directory / "policy.json").write_text(json.dumps({"proposals.environments": ["container"], **SETTINGS}))
    paths = [directory / f"{name}.proposal.json" for name in names]
    for path in paths:
        path.write_bytes(TASK_PROPOSAL.dump_json(proposal))
    (directory / "manifest.txt").write_text("\n".join(map(str, paths)))
    return paths


def argv(command: str, directory: Path) -> list[str]:
    source = directory / ("manifest.txt" if command == "run" else "test.proposal.json")
    options = {"proposals": source, "output": directory / "out", "policy": directory / "policy.json"}
    flags = [item for name, value in options.items() for item in (f"--{name}", str(value))]
    flags += ["--glm-base-url", "http://fake/v1"]
    return [command, *flags, *(["--concurrency", "2"] if command == "run" else [])]


def roots(directory: Path, sources: list[Path]) -> list[Path]:
    return [directory / "out" / driver.proposal_key(str(source)) for source in sources]


class Probe:
    """A driver process whose stdout lines arrive on a queue, so that every wait on it has a hard timeout."""

    def __init__(self, scenario: str, arguments: list[str], log: Path) -> None:
        self.log = log
        with log.open("w") as stderr:
            self.process = subprocess.Popen(
                [sys.executable, "-m", MODULE, "child", scenario, *arguments],
                cwd=ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
            )
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.seen: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def expect(self, prefix: str) -> str:
        while (line := self.lines.get(timeout=STARTUP)) is not None:
            self.seen.append(line)
            if line.startswith(prefix):
                return line
        raise AssertionError(f"exited before {prefix!r}: {self.log.read_text()[-4000:]}")

    def send(self, line: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def signal(self, stop: signal.Signals) -> int:
        """Send ``stop``; the driver's status within ``PROMPT``, once every process sharing its stdout is gone."""
        signalled = time.monotonic()
        self.process.send_signal(stop)
        try:
            code = self.process.wait(timeout=PROMPT)
        finally:
            self.process.kill()
        assert time.monotonic() - signalled < PROMPT
        try:
            while (line := self.lines.get(timeout=FOLLOW)) is not None:
                self.seen.append(line)
        except queue.Empty:
            raise AssertionError("a child outlived its driver: its stdout never reached end of file") from None
        return code


def run_to_end(scenario: str, arguments: list[str], log: Path) -> tuple[int, list[str]]:
    done = subprocess.run(
        [sys.executable, "-m", MODULE, "child", scenario, *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=STARTUP,
    )
    log.write_text(done.stderr)
    return done.returncode, done.stdout.splitlines()


def counts(lines: list[str]) -> list[dict[str, int]]:
    return [json.loads(line.removeprefix("COUNTS ")) for line in lines if line.startswith("COUNTS ")]


def relaunch(command: str, tmp_path: Path) -> list[dict[str, int]]:
    """A plain relaunch on healthy resources, which must finish; the paid calls each child made."""
    code, lines = run_to_end("resume", argv(command, tmp_path), tmp_path / "relaunch.log")
    assert code == 0, (tmp_path / "relaunch.log").read_text()[-4000:]
    return counts(lines)


def assert_reopened(tmp_path: Path, sources: list[Path], command: str) -> None:
    """Each root the crash left is claimed and not terminal, and a plain relaunch accepts every proposal. A crash
    publishes nothing after the claim, so each relaunched child buys the four author calls and two trials again."""
    assert all(published(root).head.status == Status() for root in roots(tmp_path, sources))
    assert relaunch(command, tmp_path) == [{"inference": len(script()), "trials": 2}] * len(sources)
    assert all(published_workpiece(root).terminal is Terminal.ACCEPTED for root in roots(tmp_path, sources))


# Regression: a cancelled or preempted driver job's children kept working after SIGTERM, spent their stage
# attempts on sandboxes that had died with the job and published holds only --resume-held clears; and a driver killed
# outright left its children running, buying work, as orphans. SIGTERM is now a crash, and children die with it.
@pytest.mark.timeout(300)
@pytest.mark.parametrize("stop", [signal.SIGTERM, signal.SIGKILL])
def test_a_terminated_run_takes_its_children_and_a_relaunch_finishes_both(
    proposal: TaskProposal, tmp_path: Path, stop: signal.Signals
) -> None:
    sources = inputs(proposal, tmp_path, ("first", "second"))
    unrelated = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        parent = Probe("block", argv("run", tmp_path), tmp_path / "first.log")
        parent.expect("BLOCKED")
        parent.expect("BLOCKED")
        assert parent.signal(stop) == (TERMINATED if stop is signal.SIGTERM else -signal.SIGKILL)
        assert unrelated.poll() is None, "a process outside the driver was stopped"
    finally:
        assert unrelated.stdin is not None
        unrelated.stdin.close()
        unrelated.wait(timeout=STARTUP)
    assert_reopened(tmp_path, sources, "run")


# Regression: a signal could arrive while run was starting a child or collecting one that had exited; neither
# window may leave a child behind.
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    ("scenario", "announced"),
    [
        ("block+pause-before-spawn", "SPAWNING"),
        ("block+pause-after-spawn", "SPAWNED"),
        ("first:exit0+second:block+pause-at-reap", "REAPED 0"),
    ],
)
def test_a_terminated_run_leaves_no_child_while_it_spawns_or_reaps_one(
    proposal: TaskProposal, tmp_path: Path, scenario: str, announced: str
) -> None:
    inputs(proposal, tmp_path, ("first", "second"))
    parent = Probe(scenario, argv("run", tmp_path), tmp_path / "first.log")
    parent.expect(announced)
    assert parent.signal(signal.SIGTERM) == TERMINATED


# Regression: once one child had exited, a sibling that held the parent's end of a pipe would keep its own
# watcher from ever reading end of file. The child left running dies when its parent is killed.
@pytest.mark.timeout(300)
def test_a_child_that_outlives_its_sibling_still_dies_with_the_parent(proposal: TaskProposal, tmp_path: Path) -> None:
    inputs(proposal, tmp_path, ("first", "second"))
    parent = Probe("first:exit0+second:block", argv("run", tmp_path), tmp_path / "first.log")
    parent.expect("BLOCKED")
    assert parent.signal(signal.SIGKILL) == -signal.SIGKILL


# Regression: a standalone child ignored SIGTERM while it worked. It exits 143 at once, at the first signal
# or a repeated one, and its root reopens.
@pytest.mark.timeout(300)
@pytest.mark.parametrize("signals", [1, 2])
def test_a_terminated_standalone_child_exits_and_its_root_reopens(
    proposal: TaskProposal, tmp_path: Path, signals: int
) -> None:
    sources = inputs(proposal, tmp_path)
    child = Probe("block", argv("one", tmp_path), tmp_path / "first.log")
    child.expect("BLOCKED")
    for _ in range(signals - 1):
        child.process.send_signal(signal.SIGTERM)
    assert child.signal(signal.SIGTERM) == TERMINATED
    assert_reopened(tmp_path, sources, "one")


# Regression: a forced exit that flushed output first could block on a full stdout or stderr pipe nobody read.
# A SIGTERM that landed just before the driver's write to such a pipe never interrupted it, so the Python handler never
# ran: ``uninterrupted`` puts the driver in that state on purpose.
@pytest.mark.timeout(300)
@pytest.mark.parametrize("scenario", ["flood", "flood+uninterrupted"])
def test_unread_full_output_pipes_do_not_delay_the_exit(proposal: TaskProposal, tmp_path: Path, scenario: str) -> None:
    inputs(proposal, tmp_path)
    child = subprocess.Popen(
        [sys.executable, "-m", MODULE, "child", scenario, *argv("one", tmp_path)],
        cwd=ROOT,
        stdout=subprocess.PIPE,  # read only once the child is gone
        stderr=subprocess.PIPE,  # never read
        start_new_session=True,
    )
    try:
        # The first byte on stdout means stderr is full and the driver is inside a write that cannot complete.
        assert select.select([child.stdout], [], [], STARTUP)[0], "the child never began to flood its output"
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=PROMPT) == TERMINATED
        assert child.stdout is not None and child.stdout.read(1) == b"x", "something else wrote to stdout first"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.communicate()


HIGH = 1100
"""A descriptor number above ``FD_SETSIZE`` (1024), which ``select.select`` refuses."""


# Regression: a child started after its parent had died would see a closed pipe only once its watching thread
# ran; it checks before anything else, so it claims nothing. A parent with many files open passed its pipe at fd 1030,
# which the readiness check refused, so every proposal failed before it ran.
@pytest.mark.timeout(300)
@pytest.mark.parametrize("descriptor", ["low", "high"])
@pytest.mark.parametrize("writer", ["closed", "open"])
def test_a_child_follows_its_pipe_at_any_descriptor(
    proposal: TaskProposal, tmp_path: Path, descriptor: str, writer: str
) -> None:
    sources = inputs(proposal, tmp_path)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if descriptor == "high" and hard != resource.RLIM_INFINITY and hard <= HIGH:
        pytest.skip(f"the hard open-file limit {hard} admits no descriptor numbered {HIGH}")
    raised = descriptor == "high" and soft != resource.RLIM_INFINITY and soft <= HIGH
    owned: list[int] = []  # every descriptor this test opened and still holds
    try:
        if raised:
            resource.setrlimit(resource.RLIMIT_NOFILE, (HIGH + 1, hard))
        owned.extend(os.pipe())
        lifeline, kept = owned
        if descriptor == "high":
            owned.append(lifeline := fcntl.fcntl(lifeline, fcntl.F_DUPFD, HIGH))
        opened = list(owned)
        if writer == "closed":
            os.close(kept)
            owned.remove(kept)
        code = run_to_end_with(
            "block" if writer == "closed" else "resume", lifeline, argv("one", tmp_path), tmp_path / "first.log"
        )
    finally:
        for held in owned:
            os.close(held)
        if raised:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == (soft, hard)
    assert not any(map(is_open, opened)) and (descriptor == "low" or lifeline >= HIGH)
    assert code == (TERMINATED if writer == "closed" else 0), (tmp_path / "first.log").read_text()[-4000:]
    if writer == "closed":
        assert not roots(tmp_path, sources)[0].exists()
    else:
        assert published_workpiece(roots(tmp_path, sources)[0]).terminal is Terminal.ACCEPTED


def run_to_end_with(scenario: str, lifeline: int, arguments: list[str], log: Path) -> int:
    done = subprocess.run(
        [sys.executable, "-m", MODULE, "child", scenario, *arguments, "--die-with-parent", str(lifeline)],
        cwd=ROOT,
        pass_fds=(lifeline,),
        capture_output=True,
        text=True,
        timeout=STARTUP,
    )
    log.write_text(done.stderr)
    return done.returncode


STUBBORN = """import signal, sys
from taskforge.driver import die_with_parent
die_with_parent(int(sys.argv[1]))
signal.pause()
"""
"""A child that follows only its pipe; started with SIGTERM ignored."""


def is_open(descriptor: int) -> bool:
    try:
        os.fstat(descriptor)
    except OSError:
        return False
    return True


# Regression: an embedding caller cancelled run() while it waited for a child during cleanup; both ends of the
# pipe stayed open after it returned, and the open writer kept the surviving child from ever reading end of file.
@pytest.mark.timeout(300)
async def test_a_run_cancelled_during_cleanup_closes_its_pipe_and_its_child_follows(
    proposal: TaskProposal, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, second = inputs(proposal, tmp_path, ("first", "second"))
    second.write_text("{")  # malformed: run fails with a child outstanding and enters its cleanup
    pipes: list[tuple[int, int]] = []
    pipe, popen, children, terminated = os.pipe, subprocess.Popen, [], threading.Event()

    def recorded() -> tuple[int, int]:
        pipes.append(pipe())
        return pipes[-1]

    class Stubborn:
        def __init__(self, command: list[str], **options: object) -> None:
            fd = command[command.index("--die-with-parent") + 1]
            ignore = partial(signal.signal, signal.SIGTERM, signal.SIG_IGN)
            command = [sys.executable, "-c", STUBBORN, fd]
            self.process = popen(command, cwd=ROOT, preexec_fn=ignore, **options)  # type: ignore[call-overload]
            children.append(self.process)

        def wait(self) -> int:
            return self.process.wait()

        def poll(self) -> int | None:
            return self.process.poll()

        def terminate(self) -> None:
            terminated.set()
            self.process.terminate()

    monkeypatch.setattr(os, "pipe", recorded)
    monkeypatch.setattr(driver.subprocess, "Popen", Stubborn)
    flags = dict(zip(argv("run", tmp_path)[1::2], argv("run", tmp_path)[2::2], strict=True))
    args = Namespace(
        **{name.removeprefix("--").replace("-", "_"): value for name, value in flags.items() if name != "--concurrency"},
        concurrency=2,
        buckets=4096,
        bucket_range=None,
        ranks=(),
        resume_held=False,
        glm_relay_job=None,
    )
    launch = asyncio.create_task(driver.run(args))
    try:
        assert await asyncio.to_thread(terminated.wait, STARTUP), "run never reached its cleanup"
        launch.cancel()
        with contextlib.suppress(asyncio.CancelledError, ValueError):
            await launch
        lifeline, kept = pipes[0]  # the driver's; each Popen makes its own pipes after it
        assert not is_open(lifeline) and not is_open(kept)
        assert await asyncio.to_thread(children[0].wait, STARTUP) == TERMINATED
    finally:
        for child in children:
            child.kill()


# Regression: a crash can land after a head's conditional write committed but before the driver heard back. The
# relaunch reads the committed head: a hold it skips, buying nothing, and an outage-only hold it enters and finishes.
@pytest.mark.timeout(300)
@pytest.mark.parametrize("scenario", ["held", "deferred"])
def test_a_head_committed_without_its_acknowledgement_is_what_the_relaunch_reads(
    proposal: TaskProposal, tmp_path: Path, scenario: str
) -> None:
    sources = inputs(proposal, tmp_path)
    child = Probe(f"{scenario}+lost-ack", argv("one", tmp_path), tmp_path / "first.log")
    child.expect("COMMITTED")
    assert child.signal(signal.SIGTERM) == TERMINATED
    destination = roots(tmp_path, sources)[0]
    held = published_workpiece(destination)
    assert published(destination).head.status.terminal is held.terminal is Terminal.HELD
    assert driver.deferred(held) is (scenario == "deferred")
    paid = relaunch("one", tmp_path)
    if scenario == "held":
        assert paid == [{"inference": 0, "trials": 0}]
    else:
        assert published_workpiece(destination).terminal is Terminal.ACCEPTED
        assert paid == [{"inference": len(script()), "trials": 2}]


# Control: SIGINT keeps its asyncio behaviour. The run is cancelled, sends SIGTERM to each child, joins it, and ends
# by SIGINT.
@pytest.mark.timeout(300)
def test_an_interrupted_run_stops_its_children_as_before(proposal: TaskProposal, tmp_path: Path) -> None:
    inputs(proposal, tmp_path, ("first", "second"))
    parent = Probe("block", argv("run", tmp_path), tmp_path / "first.log")
    parent.expect("BLOCKED")
    parent.expect("BLOCKED")
    assert parent.signal(signal.SIGINT) == -signal.SIGINT


def serving(monkeypatch: pytest.MonkeyPatch, bundle: Resources) -> None:
    @asynccontextmanager
    async def runtime(
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), bundle

    monkeypatch.setattr(driver, "runtime", runtime)
    monkeypatch.setenv("GLM_API_KEY", "fake")


# Control: without any signal a refusal still publishes its hold.
async def test_an_unsignalled_refusal_still_holds_the_workpiece(
    proposal: TaskProposal, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = inputs(proposal, tmp_path)
    built = resources(Envs(), ())
    serving(monkeypatch, replace(built, inference=FakeInference((Hold("the host refused the request"),))))
    flags = dict(zip(argv("one", tmp_path)[1::2], argv("one", tmp_path)[2::2], strict=True))
    args = Namespace(**{name.removeprefix("--").replace("-", "_"): value for name, value in flags.items()})
    assert await driver.one(Namespace(**vars(args), resume_held=False, glm_relay_job=None)) == 0
    held = published_workpiece(roots(tmp_path, sources)[0])
    last = held.decisions[-1]
    assert held.terminal is Terminal.HELD and isinstance(last, Incomplete)
    assert "the host refused the request" in str(last.reasons)
