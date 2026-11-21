# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One subprocess and durable replay log per proposal."""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import httpx
from pydantic import TypeAdapter
from rigging.filesystem.factory import open_url, url_to_fs
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging
from rigging.secrets import get_secret, list_secret_names, require_secret

from taskforge.checkpoint import JOURNAL_FORMAT, CheckpointSink, read_journal
from taskforge.coordinator import reasons, run_workpiece
from taskforge.effects.events import Resolved, Retrying, StageCompleted
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
from taskforge.llm.glm_provider import EndpointSource
from taskforge.model.proposal import TASK_PROPOSAL, TaskProposal, Verification
from taskforge.model.review import Incomplete
from taskforge.model.workpiece import Terminal, Workpiece
from taskforge.policy import Policy, load_policy, parse_overlay
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.report import report
from taskforge.resources.decision_model import DecisionModelHealth, DecisionModelRead
from taskforge.stages.classify import replay_rule
from taskforge.wiring import clef, endpoint_source, install, runtime

SUPERSEDED, NEEDS_MIGRATION, CORRUPT = 3, 4, 5
OUTCOMES = {
    SUPERSEDED: ("superseded", "another writer owns this root now; nothing to do"),
    NEEDS_MIGRATION: ("needs_migration", f"migrate to journal format {JOURNAL_FORMAT}, then rerun the proposal"),
    CORRUPT: ("corrupt", "move this root aside, then rerun the proposal; relaunching the shard fails it again"),
}
"""Exit codes by which a ``one`` child says why it published nothing, with what an operator does about it."""
DEFERRED, ANSWERED = 75, 10
"""``one``'s exit status for a workpiece held until the decision model answers again (EX_TEMPFAIL), and a ``--probe``
child's for one that resolved a decision-model read in its own process, the only evidence that it answers."""


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


def glm_source(policy: Policy) -> EndpointSource:
    """The GLM gateway this process reaches, from ``GLM_BASE_URL`` or ``GLM_RELAY_JOB``."""
    return endpoint_source(
        os.environ.get("GLM_BASE_URL"),
        os.environ.get("GLM_RELAY_JOB"),
        require_secret("GLM_API_KEY"),
        policy.setting("author.model", str),
        policy.setting("provider.pool", str),
    )


async def one(args: argparse.Namespace) -> int:
    """Run one proposal to a published head; ``DEFERRED`` when it waits for the decision model, else 0."""
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
    checkpoints = CheckpointSink(Path(args.work_dir) / identity, destination)
    previous = await asyncio.to_thread(checkpoints.claim, journal)
    log = EventLog(sink=checkpoints, replay_from=previous, replayable=replay_rule(previous, args.resume_held))
    wp = [Workpiece(proposal.header.id, proposal)]  # the current workpiece: each stage's start, then the returned one
    try:
        async with checkpoints.running():
            async with runtime(policy, glm_source(policy), get_secret("PARALLEL_API_KEY") or "") as (config, resources):
                wp[0] = await run_workpiece(
                    proposal, config, resources, log, install=install, reached=partial(wp.__setitem__, 0)
                )
    except* (Superseded, CorruptJournal, PublicationFailed):
        raise  # The writer stopped; it publishes neither a hold nor a final head.
    except* (Hold, Restart, Transient, LogFailure) as error:
        if checkpoints.stopped is not None:
            raise  # A stopped writer records nothing more; the failure beside the stop keeps its identity.
        wp[0] = wp[0].with_(terminal=Terminal.HELD, decisions=(*wp[0].decisions, Incomplete(reasons(error))))
        await log.record(StageCompleted(None, "", "Held", wp[0].round, Terminal.HELD))
        logging.warning("Taskforge workpiece %s held: %s", proposal.header.id, reasons(error))
    except* Exception as error:
        if checkpoints.stopped is not None:
            raise
        detail = traceback.format_exc()
        wp[0] = wp[0].with_(terminal=Terminal.HELD, decisions=(*wp[0].decisions, Incomplete((f"bug: {detail}",))))
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
    # The decision model's answers (True) and retried failures (False) in this process, in order; the latest decides.
    contact = [
        isinstance(event, Resolved)
        for record in log.records[len(previous) :]
        if isinstance(event := record.event, Resolved | Retrying) and isinstance(event.request, DecisionModelRead)
        if not (isinstance(event, Resolved) and event.replayed)
    ]
    return DEFERRED if deferred(wp[0]) else ANSWERED if args.probe and contact[-1:] == [True] else 0


async def recovery(policy: Policy) -> DecisionModelHealth:
    """HEALTHY once the default decision model's host (``one``'s gateway) says so; UNSUPPORTED after a host without
    the health route says so and ``decision_model.retry_interval`` passes, when one deferred proposal probes it."""
    async with httpx.AsyncClient() as http:
        reader = clef(policy, glm_source(policy), http)
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
                command += [
                    item
                    for name in ("output", "policy", "work_dir")
                    for item in (f"--{name.replace('_', '-')}", getattr(args, name))
                ]
                if args.resume_held:  # the operator's flag alone: a child enters an outage-only hold by itself
                    command.append("--resume-held")
                command += ["--probe"] * (path == probe)
                process = subprocess.Popen(command)
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
                if path == probe:  # only a probe's fresh answer shows the decision model reads
                    probe = None
                    if code == ANSWERED:
                        ready, parked = ready + parked, []
                if code == DEFERRED:  # a fresh outage parks what was released but has not started
                    print(f"OUTCOME deferred exit={code} proposal={path}: waits for the decision model", flush=True)
                    parked, ready = ([*parked, path], ready) if probe else ([*parked, *ready, path], [])
                elif code not in (0, ANSWERED):
                    label, action = OUTCOMES.get(code, ("failed", "see its log"))
                    labels[label] += 1
                    print(f"OUTCOME {label} exit={code} proposal={path}: {action}", flush=True)
                    if code != SUPERSEDED:
                        failed.append(path)
            if parked and waiting is None and probe is None:
                waiting = asyncio.create_task(recovery(policy))
    finally:
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


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("report")
    summary.add_argument("output_root")
    summary.add_argument("--since-hours", type=float)
    for name in ("run", "one"):
        command = commands.add_parser(name)
        command.add_argument("--proposals", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--policy", required=True)
        command.add_argument("--work-dir", default="forge-work")
        command.add_argument("--resume-held", action="store_true", help="Explicitly resume published held workpieces")
        if name == "run":
            command.add_argument("--concurrency", type=int, required=True)
            command.add_argument("--buckets", type=int, default=4096)
            command.add_argument("--bucket-range", type=lambda value: tuple(map(int, value.split(":"))))
            command.add_argument("--ranks", type=lambda value: tuple(map(int, value.split(","))), default=())
        else:
            command.add_argument("--probe", action="store_true", help="Exit ANSWERED after a fresh decision-model read")
    args = parser.parse_args()
    if args.command == "report":
        configure_coreweave_s3()
        print("\n".join(asyncio.run(report(args.output_root, args.since_hours))))
    elif args.command == "run":
        asyncio.run(run(args))
    else:
        try:
            code = asyncio.run(one(args))
        except Exception as error:
            if (code := exit_code(error)) is None:
                raise
            logging.exception("Taskforge workpiece published nothing: %s", OUTCOMES[code][0])
        sys.exit(code)


if __name__ == "__main__":
    main()
