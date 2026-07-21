# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build, validate, summarize, and upload CUDA module probe artifacts."""

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import fsspec


@dataclass(frozen=True)
class ProbeEvent:
    source: str
    raw: dict


@dataclass(frozen=True)
class ProbeSummary:
    load_count: int
    profiles: dict[str, int]
    apis: dict[str, int]
    original_results: dict[str, int]
    recovery_stages: dict[str, int]
    sync_results: dict[str, int]
    hashes: dict[str, int]
    maximum_in_flight: int


def read_probe_events(log_dir: Path) -> list[ProbeEvent]:
    """Read probe NDJSON files in stable filename and line order."""
    events = []
    for path in sorted(log_dir.glob("*.ndjson")):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed probe event at {path}:{line_number}") from error
            if not isinstance(raw, dict) or "event" not in raw:
                raise ValueError(f"Invalid probe event at {path}:{line_number}")
            events.append(ProbeEvent(source=path.name, raw=raw))
    return events


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def summarize_events(events: Iterable[ProbeEvent]) -> ProbeSummary:
    """Pair module-load events and summarize diagnostic outcomes."""
    enters: dict[tuple[str, int], dict] = {}
    profiles: Counter[str] = Counter()
    apis: Counter[str] = Counter()
    original_results: Counter[str] = Counter()
    recovery_stages: Counter[str] = Counter()
    sync_results: Counter[str] = Counter()
    hashes: Counter[str] = Counter()
    maximum_in_flight = 0
    load_count = 0
    for event in events:
        raw = event.raw
        kind = raw["event"]
        if kind not in {"load_enter", "load_exit"}:
            continue
        if "sequence" not in raw:
            raise ValueError(f"{kind} missing sequence in {event.source}")
        key = (event.source, int(raw["sequence"]))
        if kind == "load_enter":
            if key in enters:
                raise ValueError(f"Duplicate load_enter sequence {key[1]} in {event.source}")
            enters[key] = raw
            continue
        enter = enters.pop(key, None)
        if enter is None:
            raise ValueError(f"load_exit sequence {key[1]} without load_enter in {event.source}")
        load_count += 1
        profiles[str(enter.get("effective_profile", "unknown"))] += 1
        apis[str(enter.get("api", "unknown"))] += 1
        maximum_in_flight = max(maximum_in_flight, int(enter.get("in_flight", 0)))
        if "sha256" in enter:
            hashes[str(enter["sha256"])] += 1
        sync_results[str(raw.get("pre_sync_result", "missing"))] += 1
        attempts = raw.get("attempts", [])
        if attempts:
            original_results[str(attempts[0]["result"])] += 1
            successful = next((attempt["name"] for attempt in attempts if int(attempt["result"]) == 0), None)
            recovery_stages[str(successful or "failed")] += 1
        elif int(raw.get("pre_sync_result", 0)) != 0:
            recovery_stages["sync_error"] += 1
        else:
            recovery_stages["no_attempt"] += 1
    if enters:
        source, sequence = next(iter(enters))
        raise ValueError(f"load_enter sequence {sequence} without load_exit in {source}")
    return ProbeSummary(
        load_count=load_count,
        profiles=_sorted_counts(profiles),
        apis=_sorted_counts(apis),
        original_results=_sorted_counts(original_results),
        recovery_stages=_sorted_counts(recovery_stages),
        sync_results=_sorted_counts(sync_results),
        hashes=_sorted_counts(hashes),
        maximum_in_flight=maximum_in_flight,
    )


def write_summary(summary: ProbeSummary, path: Path) -> None:
    path.write_text(json.dumps(dataclasses.asdict(summary), sort_keys=True) + "\n")


def upload_probe_artifacts(log_dir: Path, prefix: str, task_index: int) -> None:
    """Upload one task's validated summary and compressed raw events."""
    events = read_probe_events(log_dir)
    summary = summarize_events(events)
    task_prefix = f"{prefix.rstrip('/')}/task-{task_index}"
    filesystem, filesystem_path = fsspec.core.url_to_fs(task_prefix)
    filesystem.makedirs(filesystem_path, exist_ok=True)
    with filesystem.open(f"{filesystem_path}/summary.json", "w") as output:
        json.dump(dataclasses.asdict(summary), output, sort_keys=True)
        output.write("\n")
    with filesystem.open(f"{filesystem_path}/events.ndjson.gz", "wb") as output:
        with gzip.GzipFile(fileobj=output, mode="wb") as compressed:
            for event in events:
                compressed.write(json.dumps(event.raw, sort_keys=True).encode() + b"\n")
    if task_index != 0:
        return
    cubins = sorted(log_dir.glob("*.cubin"))
    if not cubins:
        return
    filesystem.makedirs(f"{filesystem_path}/cubins", exist_ok=True)
    for cubin in cubins:
        with cubin.open("rb") as source, filesystem.open(f"{filesystem_path}/cubins/{cubin.name}", "wb") as output:
            shutil.copyfileobj(source, output)


def build_probe(source: Path, output: Path, compiler: str) -> dict[str, str]:
    """Compile the ABI-only preload library and return reproducibility hashes."""
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-fPIC", "-shared", str(source), "-o", str(output), "-ldl", "-pthread"],
        check=True,
    )
    return {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def check_probe_events(log_dir: Path, require_fatbinary: bool, require_elf_load: bool) -> ProbeSummary:
    events = read_probe_events(log_dir)
    summary = summarize_events(events)
    if require_fatbinary and not any(
        event.raw.get("event") == "symbol_redirect" and event.raw.get("symbol") == "cuModuleLoadFatBinary"
        for event in events
    ):
        raise ValueError("No cuModuleLoadFatBinary redirect was recorded")
    if require_elf_load and not any(
        event.raw.get("event") == "load_enter" and event.raw.get("input_kind") == "elf64" for event in events
    ):
        raise ValueError("No raw ELF module load was recorded")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--compiler", required=True)
    check = commands.add_parser("check")
    check.add_argument("--log-dir", type=Path, required=True)
    check.add_argument("--require-fatbinary", action="store_true")
    check.add_argument("--require-elf-load", action="store_true")
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--log-dir", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    upload = commands.add_parser("upload")
    upload.add_argument("--log-dir", type=Path, required=True)
    upload.add_argument("--prefix", required=True)
    upload.add_argument("--task-index", type=int, default=None)
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(arguments)
    if args.command == "build":
        print(json.dumps(build_probe(args.source, args.output, args.compiler), sort_keys=True))
        return
    if args.command == "check":
        summary = check_probe_events(args.log_dir, args.require_fatbinary, args.require_elf_load)
        print(json.dumps(dataclasses.asdict(summary), sort_keys=True))
        return
    if args.command == "summarize":
        write_summary(summarize_events(read_probe_events(args.log_dir)), args.output)
        return
    task_index = args.task_index
    if task_index is None:
        task_index = int(os.environ["IRIS_TASK_INDEX"])
    upload_probe_artifacts(args.log_dir, args.prefix, task_index)


if __name__ == "__main__":
    main()
