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
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from pydantic import TypeAdapter
from rigging.filesystem.factory import open_url, url_to_fs
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging
from rigging.secrets import get_secret, list_secret_names, require_secret

from taskforge.checkpoint import JOURNAL_FORMAT, CheckpointSink, read_journal
from taskforge.coordinator import reasons, run_workpiece
from taskforge.effects.events import StageCompleted
from taskforge.effects.issues import (
    CorruptJournal,
    Hold,
    LogFailure,
    PublicationFailed,
    Restart,
    Superseded,
    Transient,
    UnsupportedFormat,
)
from taskforge.effects.log import EventLog, Status, fold
from taskforge.model.proposal import TASK_PROPOSAL, TaskProposal
from taskforge.model.review import Incomplete
from taskforge.model.workpiece import Terminal, Workpiece
from taskforge.policy import Policy, load_policy, parse_overlay
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.report import report
from taskforge.stages.classify import replayable_grade
from taskforge.wiring import endpoint_source, install, runtime

SUPERSEDED, NEEDS_MIGRATION, CORRUPT = 3, 4, 5
OUTCOMES = {
    SUPERSEDED: ("superseded", "another writer owns this root now; nothing to do"),
    NEEDS_MIGRATION: ("needs_migration", f"migrate to journal format {JOURNAL_FORMAT}, then rerun the proposal"),
    CORRUPT: ("corrupt", "move this root aside, then rerun the proposal; relaunching the shard fails it again"),
}
"""Exit codes by which a ``one`` child says why it published nothing, with what an operator does about it."""


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
    for field, key in (("environment", "proposals.environments"), ("verification", "proposals.verifications")):
        value = getattr(proposal.header, field).value
        if value not in policy.setting(key, tuple):
            logging.info("SKIP %s: %s=%s", proposal.header.id, field, value)
            return False
    return True


async def one(args: argparse.Namespace) -> None:
    configure_coreweave_s3()
    with open_url(args.proposals, "rt") as source:
        proposal = TASK_PROPOSAL.validate_json(source.read())
    policy = run_policy(args.policy)
    if not eligible(proposal, policy):
        return
    identity = proposal_key(args.proposals)
    if identity.is_absolute() or ".." in identity.parts:
        raise ValueError("Proposal id must be a relative path without '..'")
    destination = prefix_join(args.output, str(identity))
    # Another journal format or undecodable history raises before anything is claimed or published.
    journal = await asyncio.to_thread(read_journal, destination)
    status = journal.head.status if journal else Status()
    if status.finished or (status.terminal is Terminal.HELD and not args.resume_held):
        logging.info("SKIP %s: published terminal=%s", identity, status.terminal)
        return
    checkpoints = CheckpointSink(Path(args.work_dir) / identity, destination)
    previous = await asyncio.to_thread(checkpoints.claim, journal)
    log = EventLog(sink=checkpoints, replay_from=previous, replayable=replayable_grade)
    wp = Workpiece(proposal.header.id, proposal)
    try:
        async with checkpoints.running():
            source = endpoint_source(
                os.environ.get("GLM_BASE_URL"),
                os.environ.get("GLM_RELAY_JOB"),
                require_secret("GLM_API_KEY"),
                policy.setting("author.model", str),
                policy.setting("provider.pool", str),
            )
            async with runtime(policy, source, get_secret("PARALLEL_API_KEY") or "") as (config, resources):
                wp = await run_workpiece(proposal, config, resources, log, install=install)
    except* (Superseded, CorruptJournal, PublicationFailed):
        raise  # The writer stopped; it publishes neither a hold nor a final head.
    except* (Hold, Restart, Transient, LogFailure) as error:
        if checkpoints.stopped is not None:
            raise  # A stopped writer records nothing more; the failure beside the stop keeps its identity.
        wp = wp.with_(terminal=Terminal.HELD, decisions=(*wp.decisions, Incomplete(reasons(error))))
        await log.record(StageCompleted(None, "", "Held", wp.round, Terminal.HELD))
        logging.warning("Taskforge workpiece %s held: %s", proposal.header.id, reasons(error))
    except* Exception as error:
        if checkpoints.stopped is not None:
            raise
        detail = traceback.format_exc()
        wp = wp.with_(terminal=Terminal.HELD, decisions=(*wp.decisions, Incomplete((f"bug: {detail}",))))
        await log.record(StageCompleted(None, "", "bug", wp.round, Terminal.HELD))
        logging.exception("Taskforge workpiece %s failed", proposal.header.id)
        if len(error.exceptions) == 1:
            raise error.exceptions[0] from None
        raise
    finally:
        if checkpoints.stopped is None:
            # A terminal head needs the workpiece that reached it. A cancelled run holds an earlier value, even after
            # its terminal event is durable, so it publishes a checkpoint that the next launch resumes by replay.
            status = fold(tuple(record.event for record in log.records))
            matched = wp.terminal is not None and (status.round, status.terminal) == (wp.round, wp.terminal)
            await checkpoints.finish(status if matched else Status(), wp)


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
    skipped = completed = submitted = 0
    try:
        exhausted = False
        while active or not exhausted:
            while len(active) < args.concurrency and not exhausted:
                path = next(proposals, None)
                if path is None:
                    exhausted = True
                    break
                with open_url(path, "rt") as source:
                    proposal = TASK_PROPOSAL.validate_json(source.read())
                if not eligible(proposal, policy):
                    skipped += 1
                    continue
                command = [sys.executable, "-u", "-m", "taskforge.driver", "one", "--proposals", path]
                command += [
                    item
                    for name in ("output", "policy", "work_dir")
                    for item in (f"--{name.replace('_', '-')}", getattr(args, name))
                ]
                if args.resume_held:
                    command.append("--resume-held")
                process = subprocess.Popen(command)
                submitted += 1
                active[asyncio.create_task(asyncio.to_thread(process.wait))] = (path, process)
            if active:
                done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    path, _ = active.pop(task)
                    completed += 1
                    if (code := task.result()) == 0:
                        continue
                    label, action = OUTCOMES.get(code, ("failed", "see its log"))
                    labels[label] += 1
                    print(f"OUTCOME {label} exit={code} proposal={path}: {action}", flush=True)
                    if code != SUPERSEDED:
                        failed.append(path)
    finally:
        for _, process in active.values():
            if process.poll() is None:
                process.terminate()
        await asyncio.gather(*active, return_exceptions=True)
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
    args = parser.parse_args()
    if args.command == "report":
        configure_coreweave_s3()
        print("\n".join(asyncio.run(report(args.output_root, args.since_hours))))
    elif args.command == "run":
        asyncio.run(run(args))
    else:
        try:
            asyncio.run(one(args))
        except Exception as error:
            if (code := exit_code(error)) is None:
                raise
            logging.exception("Taskforge workpiece published nothing: %s", OUTCOMES[code][0])
            sys.exit(code)


if __name__ == "__main__":
    main()
