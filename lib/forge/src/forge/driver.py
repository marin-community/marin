# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One subprocess and durable replay log per proposal."""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import httpx
from pydantic import TypeAdapter
from rigging.filesystem.factory import open_url, url_to_fs
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging

from forge.checkpoint import CheckpointSink
from forge.coordinator import reasons, run_workpiece
from forge.effects.events import StageCompleted
from forge.effects.issues import DecisionModelUnavailable, Hold, LogFailure, Restart, Transient
from forge.effects.log import EventLog, JsonlSink, Status, fold
from forge.model.proposal import TASK_PROPOSAL, TaskProposal, Verification
from forge.model.review import Incomplete
from forge.model.workpiece import WORKPIECE, Submission, Terminal, Workpiece
from forge.policy import Policy, load_policy, parse_overlay
from forge.report import report
from forge.resources.decision_model import clef_healthy
from forge.stages.classify import replayable_grade
from forge.wiring import endpoint_source, install, runtime


@dataclass(frozen=True, slots=True)
class SampleEntry:
    rank: int
    proposal: str


def proposal_key(path: str) -> PurePosixPath:
    file = PurePosixPath(urlsplit(path).path)
    return PurePosixPath(file.parent.name, file.name.removesuffix(".proposal.json").removeprefix("slot-"))


WORKPIECE_FILE = "workpiece.json"


def input_key(path: str) -> PurePosixPath:
    """A proposal's output identity; a revised workpiece keeps the identity it had in its source run root."""
    file = PurePosixPath(urlsplit(path).path)
    return (
        PurePosixPath(file.parent.parent.name, file.parent.name) if file.name == WORKPIECE_FILE else proposal_key(path)
    )


def proposal_bucket(path: str, buckets: int) -> int:
    return int.from_bytes(hashlib.sha256(str(input_key(path)).encode()).digest()) % buckets


@dataclass(frozen=True, slots=True)
class AcceptedRecord:
    """The fields ``--revise`` reads from a published workpiece of any format; the rest is ignored."""

    proposal: TaskProposal
    submission: Submission | None
    terminal: Terminal | None


def revision_inputs(root: str) -> tuple[str, ...]:
    fs, prefix = url_to_fs(root)
    paths = tuple(fs.unstrip_protocol(path) for path in sorted(fs.glob(prefix_join(prefix, f"*/*/{WORKPIECE_FILE}"))))
    if not paths:
        raise ValueError(f"No published workpieces under {root}")
    return paths


def load_input(path: str, revise: bool) -> tuple[TaskProposal, Submission | None] | None:
    """The proposal at ``path``, or with ``revise`` the proposal and accepted submission of a published workpiece.

    A workpiece that was not accepted is skipped (None); an accepted one without a submission is an error.
    """
    with open_url(path, "rt") as source:
        text = source.read()
    if not revise:
        return TASK_PROPOSAL.validate_json(text), None
    record = TypeAdapter(AcceptedRecord).validate_json(text)
    if record.terminal is not Terminal.ACCEPTED:
        logging.info("SKIP %s: terminal=%s", path, record.terminal)
        return None
    if record.submission is None:
        raise ValueError(f"Accepted workpiece {path} has no submission")
    return record.proposal, record.submission


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


SNAPSHOT_FILES = ("events.jsonl", "workpiece.json", "status.json")
"""Published in this order, so an interrupted publish leaves the stream without its status."""


def publish(workdir: Path, destination: str, names: tuple[str, ...] = SNAPSHOT_FILES) -> None:
    for name in names:
        with (workdir / name).open("rb") as source, open_url(prefix_join(destination, name), "wb") as target:
            target.write(source.read())


async def recover(workdir: Path, destination: str) -> Status | None:
    """Restore the published snapshot; the event stream, not status.json, decides whether work exists.

    Starting fresh whenever the status is missing would append a second stream numbered from 1 beside
    the published one, and the next read would splice the two.
    """
    sink = JsonlSink(prefix_join(destination, "events.jsonl"))
    try:
        snapshot = {"events.jsonl": await asyncio.to_thread(sink.read_bytes)}
    except FileNotFoundError:
        return None
    for name in SNAPSHOT_FILES[1:]:
        try:
            with open_url(prefix_join(destination, name), "rb") as source:
                snapshot[name] = source.read()
        except FileNotFoundError:
            logging.warning("Published %s lacks %s; resuming from its event stream", destination, name)
    for name, data in snapshot.items():
        (workdir / name).write_bytes(data)
    if "status.json" in snapshot:
        return TypeAdapter(Status).validate_json(snapshot["status.json"])
    return fold(tuple(record.event for record in await sink.read()))


def run_policy(path: str) -> tuple[Policy, Policy]:
    declared = load_policy(Path(__file__).parent / "policy" / "defaults")
    with open_url(path, "rt") as source:
        return declared, declared.overlaid(parse_overlay(source.read()))


def eligible(proposal: TaskProposal, policy: Policy) -> bool:
    for field, key in (("environment", "proposals.environments"), ("verification", "proposals.verifications")):
        value = getattr(proposal.header, field).value
        if value not in policy.setting(key, tuple):
            logging.info("SKIP %s: %s=%s", proposal.header.id, field, value)
            return False
    return True


DECISION_MODEL_HELD = 75
"""``one``'s exit status for a workpiece held because the decision model was unavailable (EX_TEMPFAIL)."""


def decision_model_held(wp: Workpiece) -> bool:
    """Whether ``wp`` was held because the decision model stayed unavailable (the coordinator records holds)."""
    last = wp.decisions[-1] if wp.decisions else None
    return (
        wp.terminal is Terminal.HELD
        and isinstance(last, Incomplete)
        and any(reason.startswith(f"{DecisionModelUnavailable.__name__}: ") for reason in last.reasons)
    )


async def one(args: argparse.Namespace) -> int:
    configure_coreweave_s3()
    loaded = load_input(args.proposals, args.revise)
    if loaded is None:
        return 0
    proposal, seed = loaded
    _, policy = run_policy(args.policy)
    if not eligible(proposal, policy):
        return 0
    identity = input_key(args.proposals)
    if identity.is_absolute() or ".." in identity.parts:
        raise ValueError("Proposal id must be a relative path without '..'")
    workdir = Path(args.work_dir) / identity
    workdir.mkdir(parents=True, exist_ok=True)
    destination = prefix_join(args.output, str(identity))
    sink = JsonlSink(str(workdir / "events.jsonl"))
    wp = Workpiece(proposal.header.id, proposal)
    log: EventLog | None = None
    skipped = False
    try:
        restored = await recover(workdir, destination)
        if restored is not None and (restored.finished or (restored.terminal is Terminal.HELD and not args.resume_held)):
            logging.info("SKIP %s: published terminal=%s", identity, restored.terminal)
            skipped = True
            return 0
        previous = await sink.read() if Path(sink.path).exists() else ()
        checkpoints = CheckpointSink(sink, destination, WORKPIECE.dump_json(wp))
        log = EventLog(sink=checkpoints, replay_from=previous, replayable=replayable_grade)
        async with checkpoints.running():
            source = endpoint_source(
                os.environ.get("GLM_BASE_URL"),
                os.environ.get("GLM_RELAY_JOB"),
                os.environ["GLM_API_KEY"],
                policy.setting("author.model", str),
                policy.setting("provider.pool", str),
            )
            async with runtime(policy, args.arm, source, os.environ.get("PARALLEL_API_KEY", "")) as (config, resources):
                wp = await run_workpiece(proposal, config, resources, log, install=install, seed=seed)
    except* (Hold, Restart, Transient, LogFailure) as error:
        wp = wp.with_(terminal=Terminal.HELD, decisions=(*wp.decisions, Incomplete(reasons(error))))
        if log is not None:
            await log.record(StageCompleted(None, "", "Held", wp.round, Terminal.HELD))
        logging.warning("Forge workpiece %s held: %s", proposal.header.id, reasons(error))
    except* Exception as error:
        detail = traceback.format_exc()
        wp = wp.with_(terminal=Terminal.HELD, decisions=(*wp.decisions, Incomplete((f"bug: {detail}",))))
        if log is not None:
            await log.record(StageCompleted(None, "", "bug", wp.round, Terminal.HELD))
        logging.exception("Forge workpiece %s failed", proposal.header.id)
        if len(error.exceptions) == 1:
            raise error.exceptions[0] from None
        raise
    finally:
        if not skipped:
            (workdir / "workpiece.json").write_bytes(WORKPIECE.dump_json(wp))
            # A stream that could not be read is left as published, so its parts stay repairable.
            status = (
                fold(tuple(record.event for record in log.records)) if log is not None else Status(terminal=wp.terminal)
            )
            (workdir / "status.json").write_text(json.dumps(asdict(status)))
            publish(workdir, destination, SNAPSHOT_FILES if log is not None else SNAPSHOT_FILES[1:])
    return DECISION_MODEL_HELD if decision_model_held(wp) else 0


def decision_model_health(policy: Policy) -> Callable[[], Awaitable[bool]]:
    """The gateway's Clef health read on the decision model's own route, built on first use."""

    async def healthy() -> bool:
        source = endpoint_source(
            os.environ.get("GLM_BASE_URL"),
            os.environ.get("GLM_RELAY_JOB"),
            os.environ["GLM_API_KEY"],
            policy.setting("decision_model.route", str),
            policy.setting("provider.pool", str),
        )
        async with httpx.AsyncClient() as http:
            return await clef_healthy(source, http)

    return healthy


async def run(args: argparse.Namespace) -> None:
    configure_coreweave_s3()
    start, end = args.bucket_range or (0, args.buckets)
    if args.concurrency < 1 or not 0 <= start < end <= args.buckets:
        raise ValueError("Concurrency must be positive and bucket range nonempty within buckets")
    sources = revision_inputs(args.proposals) if args.revise else inputs(args.proposals, args.ranks)
    proposals = (path for path in sources if start <= proposal_bucket(path, args.buckets) < end)
    active: dict[asyncio.Task[int], tuple[str, subprocess.Popen[bytes]]] = {}
    failures: list[str] = []
    _, policy = run_policy(args.policy)
    retry_interval = policy.setting("decision_model.retry_interval", float)
    healthy = decision_model_health(policy)
    # Decision-model proposals wait here while the decision model is down. Every retry_interval the driver reads the
    # gateway's Clef health (never a Clef read); once healthy it resumes them all after a per-driver random delay,
    # so drivers released by one Clef restart do not refill in the same second.
    deferred: list[str] = []
    down = False
    next_probe = 0.0
    resume_at: float | None = None
    skipped = completed = submitted = held = 0

    async def advance_gate() -> None:
        nonlocal down, next_probe, resume_at
        if not down:
            return
        if resume_at is None and time.monotonic() >= next_probe:
            next_probe = time.monotonic() + retry_interval
            if await healthy():
                resume_at = time.monotonic() + random.uniform(0, retry_interval)
                logging.info(
                    "Decision model healthy; resuming %d deferred proposals within %gs", len(deferred), retry_interval
                )
        if resume_at is not None and time.monotonic() >= resume_at:
            logging.info("Decision model answered again; resuming %d deferred proposals", len(deferred))
            down, resume_at = False, None

    def gate_wake() -> float | None:
        if not down:
            return None
        return max(0.0, (resume_at if resume_at is not None else next_probe) - time.monotonic())

    try:
        exhausted = False
        while active or not exhausted or deferred:
            await advance_gate()
            while len(active) < args.concurrency:
                if deferred and not down:
                    path, resume = deferred.pop(0), True
                elif not exhausted:
                    path = next(proposals, None)
                    if path is None:
                        exhausted = True
                        continue
                    loaded = load_input(path, args.revise)
                    if loaded is None or not eligible(proposal := loaded[0], policy):
                        skipped += 1
                        continue
                    if proposal.header.verification is Verification.DECISION_MODEL and down:
                        deferred.append(path)
                        continue
                    resume = args.resume_held
                else:
                    break
                command = [sys.executable, "-u", "-m", "forge.driver", "one", "--proposals", path]
                command += [
                    item
                    for name in ("arm", "output", "policy", "work_dir")
                    for item in (f"--{name.replace('_', '-')}", getattr(args, name))
                ]
                if resume:
                    command.append("--resume-held")
                if args.revise:
                    command.append("--revise")
                process = subprocess.Popen(command)
                submitted += 1
                active[asyncio.create_task(asyncio.to_thread(process.wait))] = (path, process)
            wake = gate_wake()
            if active:
                done, _ = await asyncio.wait(active, timeout=wake, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    path, _ = active.pop(task)
                    completed += 1
                    if task.result() == DECISION_MODEL_HELD:
                        held += 1
                        # Re-read health now: a hold can come from a wait that began before Clef returned.
                        down, resume_at, next_probe = True, None, time.monotonic()
                        deferred.append(path)
                        logging.warning(
                            "DECISION_MODEL_HELD %s: %d decision_model proposals deferred", path, len(deferred)
                        )
                        continue
                    if task.result() != 0:
                        failures.append(path)
            elif wake is not None:
                await asyncio.sleep(wake)
    finally:
        for _, process in active.values():
            if process.poll() is None:
                process.terminate()
        await asyncio.gather(*active, return_exceptions=True)
    print(
        f"SUMMARY submitted={submitted} completed={completed} failed={len(failures)} skipped={skipped} "
        f"decision_model_held={held}",
        flush=True,
    )
    if failures:
        raise RuntimeError(f"{len(failures)} workpieces failed: {failures}")


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
        command.add_argument("--arm", choices=("S", "T"), required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--policy", required=True)
        command.add_argument("--work-dir", default="forge-work")
        command.add_argument("--resume-held", action="store_true", help="Explicitly resume published held workpieces")
        command.add_argument(
            "--revise",
            action="store_true",
            help="--proposals is a run root: revalidate each accepted workpiece's submission into --output",
        )
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
        sys.exit(asyncio.run(one(args)))


if __name__ == "__main__":
    main()
