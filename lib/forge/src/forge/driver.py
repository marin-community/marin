# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One subprocess and durable replay log per proposal."""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from pydantic import TypeAdapter
from rigging.filesystem.factory import open_url, url_to_fs
from rigging.filesystem.s3_compat import configure_coreweave_s3

from forge.coordinator import run_workpiece
from forge.effects.events import StageCompleted
from forge.effects.log import EventLog, JsonlSink, fold
from forge.model.review import Incomplete
from forge.model.serialization import TASK_PROPOSAL, WORKPIECE
from forge.model.workpiece import Terminal, Workpiece
from forge.policy import load_policy, parse_overlay, resolution_lines
from forge.wiring import runtime


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
        paths = tuple(fs.unstrip_protocol(path) for path in sorted(fs.glob(f"{prefix.rstrip('/')}/*.proposal.json")))
    if ranks and not source.endswith("sample.json"):
        raise ValueError("Rank selection requires sample.json")
    if not paths:
        raise ValueError(f"No proposal files at {source}")
    if len({proposal_key(path) for path in paths}) != len(paths):
        raise ValueError("Proposal inputs have duplicate output identities")
    return paths


def publish(workdir: Path, destination: str) -> None:
    for name in ("events.jsonl", "workpiece.json", "status.json"):
        with (workdir / name).open("rb") as source, open_url(f"{destination}/{name}", "wb") as target:
            target.write(source.read())


async def one(args: argparse.Namespace) -> None:
    configure_coreweave_s3()
    with open_url(args.proposals, "rt") as source:
        proposal = TASK_PROPOSAL.validate_json(source.read())
    identity = proposal_key(args.proposals)
    if identity.is_absolute() or ".." in identity.parts:
        raise ValueError("Proposal id must be a relative path without '..'")
    workdir = Path(args.work_dir) / identity
    workdir.mkdir(parents=True, exist_ok=True)
    sink = JsonlSink(str(workdir / "events.jsonl"))
    previous = await sink.read() if Path(sink.path).exists() else ()
    log = EventLog(sink=sink, replay_from=previous)
    wp = Workpiece(proposal.header.id, proposal)
    try:
        declared = load_policy(Path(__file__).parent / "policy" / "defaults")
        with open_url(args.policy, "rt") as source:
            policy = declared.overlaid(parse_overlay(source.read()))
        print("\n".join(resolution_lines(declared, policy)), flush=True)
        async with runtime(policy, args.arm) as (config, resources):
            wp = await run_workpiece(proposal, config, resources, log)
    except Exception:
        detail = traceback.format_exc()
        wp = wp.with_(terminal=Terminal.HELD, decisions=(*wp.decisions, Incomplete((f"bug: {detail}",))))
        await log.record(StageCompleted("driver", "", "bug", {"round": str(wp.round), "terminal": "held"}))
        logging.exception("Forge workpiece %s failed", proposal.header.id)
        raise
    finally:
        (workdir / "workpiece.json").write_bytes(WORKPIECE.dump_json(wp))
        status = fold(tuple(record.event for record in log.records))
        (workdir / "status.json").write_text(json.dumps(asdict(status)))
        publish(workdir, f"{args.output.rstrip('/')}/{identity}")


async def run(args: argparse.Namespace) -> None:
    configure_coreweave_s3()
    if args.concurrency < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Concurrency and shard count must be positive, with shard index in range")
    proposals = iter(inputs(args.proposals, args.ranks)[args.shard_index :: args.shard_count])
    active: dict[asyncio.Task[int], tuple[str, subprocess.Popen[bytes]]] = {}
    failures: list[str] = []
    try:
        exhausted = False
        while active or not exhausted:
            while len(active) < args.concurrency and not exhausted:
                path = next(proposals, None)
                if path is None:
                    exhausted = True
                    break
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
                process = subprocess.Popen(command)
                active[asyncio.create_task(asyncio.to_thread(process.wait))] = (path, process)
            if active:
                done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    path, _ = active.pop(task)
                    if task.result() != 0:
                        failures.append(path)
    finally:
        for _, process in active.values():
            if process.poll() is None:
                process.terminate()
        await asyncio.gather(*active, return_exceptions=True)
    if failures:
        raise RuntimeError(f"{len(failures)} workpieces failed: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "one"):
        command = commands.add_parser(name)
        command.add_argument("--proposals", required=True)
        command.add_argument("--arm", choices=("S", "T"), required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--policy", required=True)
        command.add_argument("--work-dir", default="forge-work")
        if name == "run":
            command.add_argument("--concurrency", type=int, required=True)
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--shard-count", type=int, default=1)
            command.add_argument("--ranks", type=lambda value: tuple(map(int, value.split(","))), default=())
    args = parser.parse_args()
    asyncio.run(run(args) if args.command == "run" else one(args))


if __name__ == "__main__":
    main()
