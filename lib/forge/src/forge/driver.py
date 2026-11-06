# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One subprocess and durable replay log per proposal."""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from pydantic import TypeAdapter
from rigging.filesystem.factory import open_url, url_to_fs
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging

from forge.checkpoint import CheckpointSink
from forge.coordinator import reasons, run_workpiece
from forge.effects.events import StageCompleted
from forge.effects.issues import Hold, LogFailure, Restart, Transient
from forge.effects.log import EventLog, JsonlSink, Status, fold
from forge.model.proposal import TASK_PROPOSAL, TaskProposal
from forge.model.review import Incomplete
from forge.model.workpiece import WORKPIECE, Terminal, Workpiece
from forge.policy import Policy, load_policy, parse_overlay
from forge.stages.classify import replayable_grade
from forge.wiring import endpoint_source, install, runtime


@dataclass(frozen=True, slots=True)
class SampleEntry:
    rank: int
    proposal: str


def proposal_key(path: str) -> PurePosixPath:
    file = PurePosixPath(urlsplit(path).path)
    return PurePosixPath(file.parent.name, file.name.removesuffix(".proposal.json").removeprefix("slot-"))


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


def publish(workdir: Path, destination: str) -> None:
    for name in ("events.jsonl", "workpiece.json", "status.json"):
        with (workdir / name).open("rb") as source, open_url(prefix_join(destination, name), "wb") as target:
            target.write(source.read())


def recover(workdir: Path, destination: str) -> Status | None:
    """Read status last-written by publish before restoring its complete snapshot."""
    try:
        with open_url(prefix_join(destination, "status.json"), "rb") as source:
            status_bytes = source.read()
    except FileNotFoundError:
        return None
    status = TypeAdapter(Status).validate_json(status_bytes)
    snapshot = {"status.json": status_bytes}
    for name in ("events.jsonl", "workpiece.json"):
        with open_url(prefix_join(destination, name), "rb") as source:
            snapshot[name] = source.read()
    for name, data in snapshot.items():
        (workdir / name).write_bytes(data)
    return status


def run_policy(path: str) -> tuple[Policy, Policy]:
    declared = load_policy(Path(__file__).parent / "policy" / "defaults")
    with open_url(path, "rt") as source:
        return declared, declared.overlaid(parse_overlay(source.read()))


def eligible(proposal: TaskProposal, policy: Policy) -> bool:
    environment = proposal.header.environment.value
    if environment in policy.setting("proposals.environments", tuple):
        return True
    logging.info("SKIP %s: environment=%s", proposal.header.id, environment)
    return False


async def one(args: argparse.Namespace) -> None:
    configure_coreweave_s3()
    with open_url(args.proposals, "rt") as source:
        proposal = TASK_PROPOSAL.validate_json(source.read())
    _, policy = run_policy(args.policy)
    if not eligible(proposal, policy):
        return
    identity = proposal_key(args.proposals)
    if identity.is_absolute() or ".." in identity.parts:
        raise ValueError("Proposal id must be a relative path without '..'")
    workdir = Path(args.work_dir) / identity
    workdir.mkdir(parents=True, exist_ok=True)
    destination = prefix_join(args.output, str(identity))
    restored = recover(workdir, destination)
    if restored is not None and (restored.finished or (restored.terminal is Terminal.HELD and not args.resume_held)):
        logging.info("SKIP %s: published terminal=%s", identity, restored.terminal)
        return
    sink = JsonlSink(str(workdir / "events.jsonl"))
    previous = await sink.read() if Path(sink.path).exists() else ()
    wp = Workpiece(proposal.header.id, proposal)
    checkpoints = CheckpointSink(sink, destination, WORKPIECE.dump_json(wp))
    log = EventLog(sink=checkpoints, replay_from=previous, replayable=replayable_grade)
    try:
        async with checkpoints.running():
            source = endpoint_source(
                os.environ.get("GLM_BASE_URL"),
                os.environ.get("GLM_RELAY_JOB"),
                os.environ["GLM_API_KEY"],
                policy.setting("author.model", str),
            )
            async with runtime(policy, args.arm, source, os.environ.get("PARALLEL_API_KEY", "")) as (config, resources):
                wp = await run_workpiece(proposal, config, resources, log, install=install)
    except* (Hold, Restart, Transient, LogFailure) as error:
        wp = wp.with_(terminal=Terminal.HELD, decisions=(*wp.decisions, Incomplete(reasons(error))))
        await log.record(StageCompleted(None, "", "Held", wp.round, Terminal.HELD))
        logging.warning("Forge workpiece %s held: %s", proposal.header.id, reasons(error))
    except* Exception as error:
        detail = traceback.format_exc()
        wp = wp.with_(terminal=Terminal.HELD, decisions=(*wp.decisions, Incomplete((f"bug: {detail}",))))
        await log.record(StageCompleted(None, "", "bug", wp.round, Terminal.HELD))
        logging.exception("Forge workpiece %s failed", proposal.header.id)
        if len(error.exceptions) == 1:
            raise error.exceptions[0] from None
        raise
    finally:
        (workdir / "workpiece.json").write_bytes(WORKPIECE.dump_json(wp))
        status = fold(tuple(record.event for record in log.records))
        (workdir / "status.json").write_text(json.dumps(asdict(status)))
        publish(workdir, destination)


async def run(args: argparse.Namespace) -> None:
    configure_coreweave_s3()
    if args.concurrency < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Concurrency and shard count must be positive, with shard index in range")
    proposals = iter(inputs(args.proposals, args.ranks)[args.shard_index :: args.shard_count])
    active: dict[asyncio.Task[int], tuple[str, subprocess.Popen[bytes]]] = {}
    failures: list[str] = []
    _, policy = run_policy(args.policy)
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
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "forge.driver",
                    "one",
                    "--proposals",
                    path,
                    "--arm",
                    args.arm,
                    "--output",
                    args.output,
                    "--policy",
                    args.policy,
                    "--work-dir",
                    args.work_dir,
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
                    if task.result() != 0:
                        failures.append(path)
    finally:
        for _, process in active.values():
            if process.poll() is None:
                process.terminate()
        await asyncio.gather(*active, return_exceptions=True)
    print(f"SUMMARY submitted={submitted} completed={completed} failed={len(failures)} skipped={skipped}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} workpieces failed: {failures}")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "one"):
        command = commands.add_parser(name)
        command.add_argument("--proposals", required=True)
        command.add_argument("--arm", choices=("S", "T"), required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--policy", required=True)
        command.add_argument("--work-dir", default="forge-work")
        command.add_argument("--resume-held", action="store_true", help="Explicitly resume published held workpieces")
        if name == "run":
            command.add_argument("--concurrency", type=int, required=True)
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--shard-count", type=int, default=1)
            command.add_argument("--ranks", type=lambda value: tuple(map(int, value.split(","))), default=())
    args = parser.parse_args()
    asyncio.run(run(args) if args.command == "run" else one(args))


if __name__ == "__main__":
    main()
