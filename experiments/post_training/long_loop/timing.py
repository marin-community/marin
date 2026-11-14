# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Append-only worker events and overlap-aware local timing reports."""

import json
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median, quantiles
from uuid import uuid4

from rigging.filesystem.storage_path import StoragePath


@dataclass(frozen=True)
class Event:
    run: str
    stage: str
    attempt: str
    span: str
    clock: str
    kind: str
    event: str
    utc: str
    monotonic: float
    parent: str = ""
    job: str = ""
    task: str = ""
    step: int | None = None
    benchmark: str = ""
    wave: str = ""
    gpus: int = 0
    status: str = ""
    schema: int = 1


class EventLog:
    """One writer per file; merge worker files only after preserving originals."""

    def __init__(self, path: Path, run: str, stage: str, attempt: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.run = run
        self.stage = stage
        self.attempt = attempt
        self.clock = uuid4().hex

    def append(self, event: Event) -> None:
        with self.path.open("a") as output:
            output.write(json.dumps(asdict(event), sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())

    @contextmanager
    def span(self, kind: str, *, gpus: int = 0, parent: str = "", step: int | None = None) -> Iterator[str]:
        span_id = uuid4().hex

        def emit(boundary: str, status: str = "") -> None:
            self.append(
                Event(
                    run=self.run,
                    stage=self.stage,
                    attempt=self.attempt,
                    span=span_id,
                    clock=self.clock,
                    kind=kind,
                    event=boundary,
                    utc=datetime.now(UTC).isoformat(),
                    monotonic=time.monotonic(),
                    parent=parent,
                    step=step,
                    gpus=gpus,
                    status=status,
                )
            )

        emit("start")
        try:
            yield span_id
        except BaseException:
            emit("end", "failed")
            raise
        else:
            emit("end", "completed")


@contextmanager
def durable_event_log(destination: str, run: str, stage: str, attempt: str) -> Iterator[EventLog]:
    """Keep a worker journal in task outputs and upload it on normal or exceptional exit."""
    directory = Path(tempfile.mkdtemp(prefix="native-timing-", dir=os.environ.get("IRIS_OUTPUT_DIR")))
    log = EventLog(directory / "events.jsonl", run, stage, attempt)
    try:
        yield log
    finally:
        if log.path.exists():
            target = StoragePath(destination)
            target.parent.mkdirs(exist_ok=True)
            target.write_bytes(log.path.read_bytes())


def read_events(paths: list[Path]) -> list[Event]:
    return [Event(**json.loads(line)) for path in paths for line in path.read_text().splitlines() if line.strip()]


def interval_union(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def report(events: list[Event], *, synthetic: bool) -> str:
    """Report paired spans; open spans remain incomplete, never zero-duration work."""
    if len({event.run for event in events}) > 1:
        raise ValueError("Report must contain exactly one run")
    boundaries: dict[tuple[str, str, str, str], dict[str, Event]] = defaultdict(dict)
    for event in events:
        key = (event.stage, event.attempt, event.clock, event.span)
        if event.event not in ("start", "end") or event.event in boundaries[key]:
            raise ValueError(f"Invalid or duplicate boundary: {key}")
        boundaries[key][event.event] = event
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    allocations: list[tuple[float, int]] = []
    gpu_hours = 0.0
    incomplete = 0
    failed = 0
    utc_values = [datetime.fromisoformat(event.utc).timestamp() for event in events]
    for pair in boundaries.values():
        if set(pair) != {"start", "end"}:
            incomplete += 1
            continue
        start, end = pair["start"], pair["end"]
        duration = end.monotonic - start.monotonic
        if duration < 0 or start.kind != end.kind or start.gpus != end.gpus:
            raise ValueError("Inconsistent span boundaries")
        groups[(start.stage, start.kind)].append(duration)
        begin, stop = (datetime.fromisoformat(event.utc).timestamp() for event in (start, end))
        if stop < begin:
            raise ValueError("UTC clock moved backwards")
        intervals[start.kind].append((begin, stop))
        failed += end.status == "failed"
        if start.kind == "allocation":
            allocations.extend(((begin, start.gpus), (stop, -start.gpus)))
            gpu_hours += duration * start.gpus / 3600
    allocated = peak = 0
    for _, change in sorted(allocations):
        allocated += change
        peak = max(peak, allocated)
    elapsed = max(utc_values) - min(utc_values) if utc_values else 0
    lines = [
        "# Long-loop timing report",
        "",
        "SYNTHETIC LOCAL EVENTS — not cluster measurements." if synthetic else "Recorded runtime events.",
        "",
        f"Observed event envelope: {elapsed:.3f} seconds (not proof of run completion).",
        f"Incomplete spans: {incomplete}; failed spans: {failed}.",
        (
            f"Paired allocation GPU-hours: {gpu_hours:.3f}; observed peak: {peak} GPUs."
            if allocations
            else "Allocation GPU-hours and peak: unavailable (no allocation boundaries captured)."
        ),
        "Allocation totals exclude incomplete spans; a missing allocation boundary prevents cap certification.",
        "",
    ]
    components = {
        "Queue and allocation": ("queue", "allocation"),
        "Model loading, server startup and compilation": ("startup", "spin", "load", "compil"),
        "File transfers and exports": ("transfer", "export", "convert"),
        "SFT and RL computation": ("train", "learner", "rollout", "optimizer", "weight_sync"),
        "Checkpoint saves and drain": ("checkpoint", "drain"),
        "Validation and evaluation": ("eval", "judge", "benchmark"),
        "Archive and cleanup": ("archive", "cleanup"),
    }
    remaining = dict(groups)
    for title, keywords in components.items():
        selected = {
            key: values
            for key, values in remaining.items()
            if "including_queue" not in key[1] and any(word in key[1] for word in keywords)
        }
        lines.extend(["", f"## {title}", ""])
        if not selected:
            lines.append("No distinct phase measurements captured; unavailable, not zero.")
            continue
        lines.extend(["| Stage | Kind | Spans | Work seconds | p50 | p90 | max |", "|---|---|---:|---:|---:|---:|---:|"])
        for (stage, kind), durations in sorted(selected.items()):
            p90 = quantiles(durations, n=10, method="inclusive")[8] if len(durations) > 1 else durations[0]
            lines.append(
                f"| {stage} | {kind} | {len(durations)} | {sum(durations):.3f} | "
                f"{median(durations):.3f} | {p90:.3f} | {max(durations):.3f} |"
            )
            del remaining[(stage, kind)]
    lines.extend(
        [
            "",
            "## Inclusive stage spans and other measurements",
            "",
            "These spans can include the components above; do not add them to component totals.",
            "",
        ]
    )
    for (stage, kind), durations in sorted(remaining.items()):
        lines.append(f"- {stage} / {kind}: {sum(durations):.3f} seconds across {len(durations)} spans.")
    lines.extend(["", "UTC interval unions (overlapping work is counted once):", ""])
    lines.extend(f"- {kind}: {interval_union(spans):.3f} seconds" for kind, spans in sorted(intervals.items()))
    lines.extend(
        [
            "",
            "Durations use same-clock monotonic end - start. UTC unions and allocation overlap assume",
            "synchronized host clocks; cross-host skew is unmeasured. Work sums are not wall time.",
            "Queue remains in elapsed time. Absent phase instrumentation is unavailable, not zero.",
            "Retries remain separate by attempt ID. Raw JSONL is the source of every table.",
            "",
        ]
    )
    return "\n".join(lines)
