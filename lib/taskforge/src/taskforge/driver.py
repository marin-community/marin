# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Taskforge proposals: ``run`` a shard, each proposal in its own ``one`` process; ``report`` an output root.

``one`` exits 0 once it publishes a head or skips the proposal; 3 when another writer owns the root; 4 when the root
has another journal format (rerun at a fresh output prefix; no migration ships); 5 when the root's journal is corrupt; 75
when the workpiece is held until the decision model answers again; with ``--probe``, 10 when its own decision-model
reader's last outcome was a valid answer; and 1 on any other failure. ``run`` consumes a probe's 10 silently,
releasing what it parked; it prints an OUTCOME line for every other nonzero child (a 10 from any child but the probe
is a failure) and fails when any child exited other than 0, 3 or 75. SIGTERM is a crash: the process exits 143 at
once from the moment ``main`` has installed its handler (one SIGTERM while the interpreter still imports is lost when
the driver is PID 1), and each ``run`` child, which reads end of file from its parent's pipe, follows; a relaunch
reopens each root's last committed head. ``main`` is the process entry point, not an embedding API: it leaves its
SIGTERM handler installed. ``one`` and ``run`` install nothing.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import select
import signal
import subprocess
import sys
import threading
import traceback
from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import httpx
from pydantic import TypeAdapter
from rigging.filesystem.factory import open_url, url_to_fs
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging
from rigging.secrets import get_secret, list_secret_names, require_secret

from taskforge.checkpoint import CheckpointSink, read_journal
from taskforge.coordinator import reasons, run_workpiece
from taskforge.effects.events import StageCompleted
from taskforge.effects.issues import (
    CorruptJournal,
    DecisionModelUnavailable,
    Hold,
    LogFailure,
    PublicationFailed,
    Restart,
    Superseded,
    Transient,
    UnsupportedFormat,
)
from taskforge.effects.log import EventLog, Status, fold
from taskforge.llm.glm_provider import API_ROOT, EndpointSource
from taskforge.model.proposal import TASK_PROPOSAL, TaskProposal, Verification
from taskforge.model.review import Incomplete
from taskforge.model.workpiece import Terminal, Workpiece
from taskforge.policy import Policy, load_policy, parse_overlay
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.report import report
from taskforge.resources.decision_model import DecisionModel, DecisionModelHealth
from taskforge.stages.classify import replaced_refusals, replayable_grade
from taskforge.wiring import clef, endpoint_source, install, runtime

SUPERSEDED, NEEDS_MIGRATION, CORRUPT = 3, 4, 5
OUTCOMES = {
    SUPERSEDED: ("superseded", "another writer owns this root now; nothing to do"),
    NEEDS_MIGRATION: ("needs_migration", "rerun at a fresh output prefix (no migration ships in this repository)"),
    CORRUPT: ("corrupt", "move this root aside, then rerun the proposal; relaunching the shard fails it again"),
}
"""Exit codes by which a ``one`` child says why it published nothing, with what an operator does about it."""
DEFERRED, ANSWERED = 75, 10
"""``one``'s exit status for a workpiece held until the decision model answers again (EX_TEMPFAIL), and a ``--probe``
child's when its own reader's last physical outcome was a valid answer, the only evidence that the model answers."""


def deferred(wp: Workpiece | None) -> bool:
    """Whether ``wp`` is held only until the decision model answers, so that a launch enters it unasked."""
    last = wp.decisions[-1] if wp is not None and wp.decisions else None
    awaits = last.awaits if isinstance(last, Incomplete) else None
    return wp is not None and wp.terminal is Terminal.HELD and awaits == DecisionModelUnavailable.awaits


@dataclass(frozen=True, slots=True)
class SampleEntry:
    rank: int
    proposal: str


def proposal_key(path: str) -> PurePosixPath:
    file = PurePosixPath(urlsplit(path).path)
    return PurePosixPath(file.parent.name, file.name.removesuffix(".proposal.json").removeprefix("slot-"))


def proposal_bucket(path: str, buckets: int) -> int:
    return int.from_bytes(hashlib.sha256(str(proposal_key(path)).encode()).digest()) % buckets


def inputs(source: str, ranks: tuple[int, ...] = ()) -> tuple[str, ...]:
    fs, prefix = url_to_fs(source)
    if source.endswith(".proposal.json"):
        paths = (source,)
    elif source.endswith(".json"):
        with open_url(source, "rt") as stream:
            sample = TypeAdapter(tuple[SampleEntry, ...]).validate_python(json.load(stream)["sample"])
        if set(ranks) - {entry.rank for entry in sample}:
            raise ValueError("Requested ranks are missing from the sample")
        paths = tuple(entry.proposal for entry in sample if not ranks or entry.rank in ranks)
    elif fs.isfile(prefix):
        with open_url(source, "rt") as stream:
            paths = tuple(line.strip() for line in stream if line.strip())
    else:
        paths = tuple(fs.unstrip_protocol(path) for path in sorted(fs.glob(prefix_join(prefix, "*.proposal.json"))))
    if ranks and not source.endswith("sample.json"):
        raise ValueError("Rank selection requires sample.json")
    if not paths:
        raise ValueError(f"No proposal files at {source}")
    if len({proposal_key(path) for path in paths}) != len(paths):
        raise ValueError("Proposal inputs have duplicate output identities")
    return paths


def run_policy(path: str) -> Policy:
    with open_url(path, "rt") as source:
        return load_policy(DEFAULTS_DIR).overlaid(parse_overlay(source.read()))


def eligible(proposal: TaskProposal, policy: Policy) -> bool:
    for name, key in (("environment", "proposals.environments"), ("verification", "proposals.verifications")):
        value = getattr(proposal.header, name).value
        if value not in policy.setting(key, tuple):
            logging.info("SKIP %s: %s=%s", proposal.header.id, name, value)
            return False
    return True


def glm_source(policy: Policy, args: argparse.Namespace) -> EndpointSource:
    """The GLM gateway named by ``--glm-base-url`` or ``--glm-relay-job``."""
    return endpoint_source(
        args.glm_base_url,
        args.glm_relay_job,
        require_secret("GLM_API_KEY"),
        policy.setting("author.model", str),
        policy.setting("provider.pool", str),
    )


async def one(args: argparse.Namespace, controller_url: str | None = None) -> int:
    """Run one proposal to a published head; ``DEFERRED`` when it waits for the decision model, ``ANSWERED`` when a
    ``--probe`` run got a fresh decision-model answer, else 0."""
    configure_coreweave_s3()
    with open_url(args.proposals, "rt") as source:
        proposal = TASK_PROPOSAL.validate_json(source.read())
    policy = run_policy(args.policy)
    if not eligible(proposal, policy):
        return 0
    identity = proposal_key(args.proposals)
    if identity.is_absolute() or ".." in identity.parts:
        raise ValueError("Proposal id must be a relative path without '..'")
    destination = prefix_join(args.output, str(identity))
    # Another journal format or undecodable history raises before anything is claimed or published.
    journal = await asyncio.to_thread(read_journal, destination)
    status = journal.head.status if journal else Status()
    held = status.terminal is Terminal.HELD
    enter = args.resume_held or (held and journal is not None and deferred(await asyncio.to_thread(journal.workpiece)))
    if status.finished or (held and not enter):
        logging.info("SKIP %s: published terminal=%s", identity, status.terminal)
        return 0
    checkpoints = CheckpointSink(destination)
    previous = await asyncio.to_thread(checkpoints.claim, journal)
    replaces = replaced_refusals(previous, args.resume_held)
    log = EventLog(sink=checkpoints, replay_from=previous, replayable=replayable_grade, replaces=replaces)
    wp = [Workpiece(proposal.header.id, proposal)]  # the current workpiece: each stage's start, then the returned one
    readers: list[DecisionModel | None] = [None]  # this process's decision model, as the runtime built it
    try:
        async with checkpoints.running():
            gateway, key = glm_source(policy, args), get_secret("PARALLEL_API_KEY") or ""
            async with runtime(policy, gateway, key, controller_url=controller_url) as (config, resources):
                readers[0] = resources.decision_model
                wp[0] = await run_workpiece(
                    proposal, config, resources, log, install=install, reached=partial(wp.__setitem__, 0)
                )
    except* (Superseded, CorruptJournal, PublicationFailed):
        raise  # The writer stopped; it publishes neither a hold nor a final head.
    except* (Hold, Restart, Transient, LogFailure) as error:
        if checkpoints.stopped is not None:
            raise  # A stopped writer records nothing more; the failure beside the stop keeps its identity.
        wp[0] = replace(wp[0], terminal=Terminal.HELD, decisions=(*wp[0].decisions, Incomplete(reasons(error))))
        await log.record(StageCompleted(None, "", "Held", wp[0].round, Terminal.HELD))
        logging.warning("Taskforge workpiece %s held: %s", proposal.header.id, reasons(error))
    except* Exception as error:
        if checkpoints.stopped is not None:
            raise
        detail = traceback.format_exc()
        wp[0] = replace(wp[0], terminal=Terminal.HELD, decisions=(*wp[0].decisions, Incomplete((f"bug: {detail}",))))
        await log.record(StageCompleted(None, "", "bug", wp[0].round, Terminal.HELD))
        logging.exception("Taskforge workpiece %s failed", proposal.header.id)
        if len(error.exceptions) == 1:
            raise error.exceptions[0] from None
        raise
    finally:
        if checkpoints.stopped is None:
            # A terminal head needs the workpiece that reached it. A cancelled run holds its last stage's start, even
            # after its terminal event is durable, so it publishes a checkpoint that the next launch resumes by replay.
            status = fold(tuple(record.event for record in log.records))
            matched = wp[0].terminal is not None and (status.round, status.terminal) == (wp[0].round, wp[0].terminal)
            await checkpoints.finish(status if matched else Status(), wp[0])
    # A probe answers only when its own reader's last physical outcome is an answer; in-process callers omit the flag.
    reader = readers[0] if getattr(args, "probe", False) else None
    return DEFERRED if deferred(wp[0]) else ANSWERED if reader is not None and reader.answered else 0


async def recovery(policy: Policy, args: argparse.Namespace) -> DecisionModelHealth:
    """HEALTHY once the default decision model's host (``one``'s gateway) says so; UNSUPPORTED after a host without
    the health route says so and ``decision_model.retry_interval`` passes, when one deferred proposal probes it."""
    async with httpx.AsyncClient() as http:
        reader = clef(policy, glm_source(policy, args), http)
        while (health := await reader.health()) is not DecisionModelHealth.HEALTHY:
            await asyncio.sleep(policy.setting("decision_model.retry_interval", float))
            if health is DecisionModelHealth.UNSUPPORTED:
                break
    return health


async def run(args: argparse.Namespace) -> None:
    configure_coreweave_s3()
    logging.info("Secret files (names only): %s", ", ".join(list_secret_names()) or "none")
    start, end = args.bucket_range or (0, args.buckets)
    if args.concurrency < 1 or not 0 <= start < end <= args.buckets:
        raise ValueError("Concurrency must be positive and bucket range nonempty within buckets")
    proposals = (
        path for path in inputs(args.proposals, args.ranks) if start <= proposal_bucket(path, args.buckets) < end
    )
    active: dict[asyncio.Task[int], tuple[str, subprocess.Popen[bytes]]] = {}
    labels, failed = Counter[str](), list[str]()
    policy = run_policy(args.policy)
    # Decision-model proposals wait in ``parked`` while one background ``waiting`` health read runs, or while the one
    # ``probe`` of a host without the route runs; ``ready`` is what a recovery released, launched first.
    parked, ready = list[str](), list[str]()
    waiting: asyncio.Task[DecisionModelHealth] | None = None
    probe: str | None = None
    skipped = completed = submitted = 0
    # Every child reads end of file from ``lifeline`` once this process is gone, however it died; only it holds ``kept``.
    lifeline, kept = os.pipe()
    try:
        exhausted = False
        while active or not exhausted or parked or ready:
            while len(active) < args.concurrency:
                if ready:
                    path = ready.pop(0)
                elif exhausted or (path := next(proposals, None)) is None:
                    exhausted = True
                    break
                else:
                    with open_url(path, "rt") as source:
                        proposal = TASK_PROPOSAL.validate_json(source.read())
                    if not eligible(proposal, policy):
                        skipped += 1
                        continue
                    if (waiting or probe) and proposal.header.verification is Verification.DECISION_MODEL:
                        parked.append(path)
                        continue
                command = [sys.executable, "-u", "-m", "taskforge.driver", "one", "--proposals", path]
                command += ["--die-with-parent", str(lifeline)]
                command += [
                    item
                    for name in ("output", "policy", "glm_base_url", "glm_relay_job")
                    if getattr(args, name) is not None
                    for item in (f"--{name.replace('_', '-')}", getattr(args, name))
                ]
                if args.resume_held:  # the operator's flag alone: a child enters an outage-only hold by itself
                    command.append("--resume-held")
                command += ["--probe"] * (path == probe)
                process = subprocess.Popen(command, pass_fds=(lifeline,))
                submitted += 1
                active[asyncio.create_task(asyncio.to_thread(process.wait))] = (path, process)
            if not active and waiting is None:
                break  # the manifest ran out during admission, with nothing parked
            done, _ = await asyncio.wait([*active, *([waiting] if waiting else [])], return_when=asyncio.FIRST_COMPLETED)
            if waiting is not None and waiting in done:
                if waiting.result() is DecisionModelHealth.HEALTHY:
                    ready, parked = ready + parked, []
                else:
                    ready.append(probe := parked.pop(0))
                waiting = None
            for task in [task for task in active if task in done]:
                path, _ = active.pop(task)
                completed += 1
                code = task.result()
                if path == probe:  # only a probe's fresh answer shows the decision model reads; from others, a failure
                    probe = None
                    if code == ANSWERED:
                        ready, parked, code = ready + parked, [], 0
                if code == DEFERRED:  # a fresh outage parks what was released but has not started
                    print(f"OUTCOME deferred exit={code} proposal={path}: waits for the decision model", flush=True)
                    parked, ready = ([*parked, path], ready) if probe else ([*parked, *ready, path], [])
                elif code != 0:
                    label, action = OUTCOMES.get(code, ("failed", "see its log"))
                    labels[label] += 1
                    print(f"OUTCOME {label} exit={code} proposal={path}: {action}", flush=True)
                    if code != SUPERSEDED:
                        failed.append(path)
            if parked and waiting is None and probe is None:
                waiting = asyncio.create_task(recovery(policy, args))
    finally:
        os.close(lifeline)  # first, so that neither end outlives this call and each child reads end of file
        os.close(kept)
        if waiting is not None:
            waiting.cancel()
        for _, process in active.values():
            if process.poll() is None:
                process.terminate()
        await asyncio.gather(*active, *([waiting] if waiting else []), return_exceptions=True)
    outcomes = " ".join(f"{label}={labels[label]}" for label, _ in OUTCOMES.values())
    print(
        f"SUMMARY submitted={submitted} completed={completed} failed={len(failed)} skipped={skipped} {outcomes}",
        flush=True,
    )
    if failed:
        raise RuntimeError(f"{len(failed)} workpieces failed: {failed}")


def exit_code(error: Exception) -> int | None:
    """The outcome code when every leaf of ``error`` is that one outcome; anything mixed is an ordinary failure."""
    group = error if isinstance(error, ExceptionGroup) else ExceptionGroup("one", [error])
    for code, issue in ((SUPERSEDED, Superseded), (NEEDS_MIGRATION, UnsupportedFormat), (CORRUPT, CorruptJournal)):
        if group.split(issue)[1] is None:
            return code
    return None


def relay_job(value: str) -> str:
    """A non-blank relay job, so that a quoted unset variable is a usage error before anything is claimed."""
    if not value.strip():
        raise argparse.ArgumentTypeError(f"must name an Iris relay job, got {value!r}")
    return value


def base_url(value: str) -> str:
    if not value.endswith(API_ROOT):
        raise argparse.ArgumentTypeError(f"must be a URL ending in {API_ROOT}, got {value!r}")
    return value


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("report")
    summary.add_argument("output_root")
    summary.add_argument("--since-hours", type=float)
    for name in ("run", "one"):
        command = commands.add_parser(name)
        command.add_argument("--proposals", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--policy", required=True)
        gateway = command.add_mutually_exclusive_group(required=True)
        gateway.add_argument("--glm-relay-job", type=relay_job, help="The Iris relay job that resolves the GLM gateway")
        gateway.add_argument("--glm-base-url", type=base_url, help=f"A direct GLM gateway endpoint ending in {API_ROOT}")
        command.add_argument(
            "--resume-held",
            action="store_true",
            help="Also resume published held workpieces; one held only for the decision model resumes without it",
        )
        if name == "run":
            command.add_argument("--concurrency", type=int, required=True)
            command.add_argument("--buckets", type=int, default=4096)
            command.add_argument("--bucket-range", type=lambda value: tuple(map(int, value.split(":"))))
            command.add_argument("--ranks", type=lambda value: tuple(map(int, value.split(","))), default=())
        else:
            command.add_argument("--probe", action="store_true", help="Exit 10 after a fresh decision-model answer")
            command.add_argument("--die-with-parent", type=int, metavar="FD", help="Exit 143 at end of file on FD")
    return parser


def die_with_parent(lifeline: int) -> None:
    """Exit 143 once ``lifeline``, a pipe the parent holds open and never writes, reads end of file: at once when the
    parent is already gone, else from a watching thread."""
    (readable := select.poll()).register(lifeline, select.POLLIN)  # poll, unlike select, takes any descriptor
    if readable.poll(0):
        os._exit(143)
    threading.Thread(target=lambda: os.read(lifeline, 1) or os._exit(143), daemon=True).start()


def exit_on_sigterm() -> None:
    """Exit 143 at SIGTERM: a crash, so the journal's last committed head survives. A Python handler runs only once the
    main thread returns to the interpreter, which a write to a full pipe nobody reads never does when the signal landed
    just before it; the C handler also writes the signal's number to the wakeup pipe, and a watching thread exits."""
    signal.signal(signal.SIGTERM, lambda *_: os._exit(143))
    wakeup, woken = os.pipe()
    os.set_blocking(woken, False)
    signal.set_wakeup_fd(woken)

    def watch() -> None:
        while number := os.read(wakeup, 1):
            if number[0] == signal.SIGTERM:
                os._exit(143)

    threading.Thread(target=watch, daemon=True).start()


def main() -> None:
    args = cli().parse_args()
    if threading.current_thread() is threading.main_thread():  # an in-process test child runs on another thread
        exit_on_sigterm()
        if getattr(args, "die_with_parent", None) is not None:
            die_with_parent(args.die_with_parent)
    configure_logging()
    if args.command == "report":
        configure_coreweave_s3()
        print("\n".join(asyncio.run(report(args.output_root, args.since_hours))))
    elif args.command == "run":
        asyncio.run(run(args))
    else:
        try:
            # Iris sets the controller address inside a job; outside one it is the caller's to export.
            code = asyncio.run(one(args, os.environ.get("IRIS_CONTROLLER_URL")))
        except Exception as error:
            if (code := exit_code(error)) is None:
                raise
            logging.exception("Taskforge workpiece published nothing: %s", OUTCOMES[code][0])
        sys.exit(code)


if __name__ == "__main__":
    main()
