# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A driver process for the termination tests: ``driver.main`` on fake resources, signalled by the test.

``child SCENARIO ARGV...`` runs ``taskforge.driver ARGV`` (``one`` or ``run``). SCENARIO joins tokens with ``+``; a
``NAME:token`` applies only to the proposal ``NAME``, and a ``run`` launches each child through this module with the
plain tokens and its own. A ``one`` that returns prints ``COUNTS`` with the paid calls it made.

A child's tokens:

- ``block``: the author's four calls resolve, then both validation trials print ``BLOCKED <pid>`` and wait forever.
- ``held``: the author's first call is refused (``Hold``); ``deferred``: the decision model is unavailable instead.
- ``lost-ack``: once a terminal head is written, the writer prints ``COMMITTED`` and never returns.
- ``flood``: once logging is set up, it fills stderr, then writes to stdout without end, each write larger than any
  pipe: its stdout's first byte comes from inside a write that cannot complete. Nothing reaches stdout before.
- ``uninterrupted``: with ``flood``, the main thread blocks SIGTERM first, so the signal never interrupts its write and
  never reaches the Python handler: what a signal landing just before the write would do.
- ``resume``: runs to the end on healthy resources.
- ``exit<N>``: exits N at once, without the driver.

A ``run``'s tokens: ``pause-before-spawn`` and ``pause-after-spawn`` (its first launch prints ``SPAWNING``, or
``SPAWNED <pid>`` once the child exists, and waits for a line on stdin), and ``pause-at-reap`` (a child's exit status
prints ``REAPED <code>`` and waits for a line on stdin before it reaches the driver).
"""

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, override

import pytest
from lib.taskforge.tests.resource_doubles import FakeInference
from lib.taskforge.tests.stage_fakes import SolverEngine
from lib.taskforge.tests.test_end_to_end import Envs, resources, script
from rolloutengine.models import GradedRollout, Model, TaskRolloutSpec

from taskforge import checkpoint, driver
from taskforge.config import TaskforgeConfig
from taskforge.effects.issues import DecisionModelUnavailable, Hold
from taskforge.llm.glm_provider import EndpointSource
from taskforge.policy import Policy
from taskforge.resources.bundle import Resources
from taskforge.wiring import build_config

MODULE = "lib.taskforge.tests.termination_probe"


def announce(line: str) -> None:
    sys.stdout.write(line + "\n")  # one write: siblings share the pipe
    sys.stdout.flush()


class Blocking(SolverEngine):
    """Both trials start, announce it once, and wait."""

    def __init__(self) -> None:
        super().__init__(2)
        self.blocked = asyncio.Event()

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        self.started.append(str(run_id))
        if len(self.started) == self.count:
            self.blocked.set()
            announce(f"BLOCKED {os.getpid()}")
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def bundle(tokens: set[str]) -> Resources:
    built = resources(Envs(), script())
    if "held" in tokens:
        return replace(built, inference=FakeInference((Hold("the host refused the request"),)))
    if "deferred" in tokens:
        return replace(built, inference=FakeInference((DecisionModelUnavailable("the decision model is down"),)))
    if "block" in tokens:
        return replace(built, rollouts=replace(built.rollouts, engine=Blocking()))
    return built


def seams(patch: pytest.MonkeyPatch, tokens: set[str]) -> None:
    """The child's own seams: a lost acknowledgement and a flood of output."""
    swap, configure = checkpoint.CheckpointSink._swap, driver.configure_logging

    def committing(sink: checkpoint.CheckpointSink, head: checkpoint.Head, expected: str | None) -> None:
        swap(sink, head, expected)
        if "lost-ack" in tokens and head.status.terminal is not None:
            announce("COMMITTED")
            threading.Event().wait()  # the acknowledgement never reaches the driver

    def flooding() -> None:
        configure()
        if "flood" not in tokens:
            return
        if "uninterrupted" in tokens:
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        os.set_blocking(2, False)
        with contextlib.suppress(BlockingIOError):
            while True:
                os.write(2, b"x" * 2**16)
        os.set_blocking(2, True)
        while True:
            os.write(1, b"x" * 2**24)  # no pipe holds 16 MiB

    patch.setattr(checkpoint.CheckpointSink, "_swap", committing)
    patch.setattr(driver, "configure_logging", flooding)


class Reaped:
    """A child whose exit status reaches the driver only after the test answers on stdin."""

    def __init__(self, process: subprocess.Popen[Any]) -> None:
        self.process = process

    def wait(self) -> int:
        code = self.process.wait()
        announce(f"REAPED {code}")
        sys.stdin.readline()
        return code

    def poll(self) -> int | None:
        return self.process.poll()

    def terminate(self) -> None:
        self.process.terminate()


def parent(patch: pytest.MonkeyPatch, scenario: str) -> None:
    """Route each child through this module, and apply the run's own seams."""
    tokens = scenario.split("+")
    plain = [token for token in tokens if ":" not in token and not token.startswith("pause-")]
    named = [token.split(":", 1) for token in tokens if ":" in token]
    launch, launched = subprocess.Popen, []

    def pause(token: str, line: str) -> None:
        if token in tokens and len(launched) == 1:
            announce(line)
            sys.stdin.readline()

    def through_probe(command: list[str], **options: Any) -> Any:
        name = Path(command[command.index("--proposals") + 1]).name.removesuffix(".proposal.json")
        start = command.index("taskforge.driver")
        child = "+".join([*plain, *(token for owner, token in named if owner == name)])
        launched.append(name)
        pause("pause-before-spawn", "SPAWNING")
        process = launch([*command[: start - 1], "-m", MODULE, "child", child, *command[start + 1 :]], **options)
        pause("pause-after-spawn", f"SPAWNED {process.pid}")
        return Reaped(process) if "pause-at-reap" in tokens else process

    patch.setattr(driver.subprocess, "Popen", through_probe)


def main(scenario: str, argv: list[str]) -> None:
    tokens = {token for token in scenario.split("+") if ":" not in token}
    for token in tokens:
        if token.removeprefix("exit").isdigit():
            sys.exit(int(token.removeprefix("exit")))
    os.environ.update(GLM_API_KEY="fake")
    built = bundle(tokens)

    @asynccontextmanager
    async def runtime(
        policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), built

    patch = pytest.MonkeyPatch()  # the process ends with the driver, so nothing is undone
    patch.setattr(driver, "runtime", runtime)
    # Only the claim and the final publication write the head, so that a kill never meets a head write in progress.
    patch.setattr(checkpoint, "CHECKPOINT_EVENTS", 10**6)
    seams(patch, tokens)
    if argv[0] == "run":
        parent(patch, scenario)
    sys.argv = ["driver", *argv]
    try:
        driver.main()
    finally:
        engine, inference = built.rollouts.engine, built.inference
        if argv[0] == "one" and isinstance(engine, SolverEngine) and isinstance(inference, FakeInference):
            announce(f"COUNTS {json.dumps({'inference': len(inference.requests), 'trials': len(engine.started)})}")


if __name__ == "__main__":
    assert sys.argv[1] == "child"
    main(sys.argv[2], sys.argv[3:])
