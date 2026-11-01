# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A compact per-step summary of a training profile, for MFU work.

Reads one profile session (XPlane protobuf, or a Perfetto/Chrome trace) and reports, per training step and averaged
over the profiled devices: the top kernels and their share of the step, compute versus communication time, the
communication left exposed (not overlapped by compute) per collective type, host-offload copies (device-to-host and
host-to-device) and how much of them is exposed, and the step time. Peak memory comes from the run itself (device
memory stats), since a trace does not carry it.

    python -m marin.profiling.step_profile <run id> [--compare <run id>]

prints the summary of a run's uploaded profile (from the run's XProf directory under the cluster's temp bucket), and
with ``--compare`` a per-kernel diff against a second run.
"""

import argparse
import bisect
import dataclasses
import logging
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from levanter.callbacks.profiler import XprofUploadConfig
from rigging.filesystem.storage_path import StoragePath

from marin.profiling.ingest import find_profile_trace
from marin.profiling.trace_summary import (
    BreakdownMode,
    TraceEvent,
    TraceEventTrack,
    TraceTrackAggregate,
    collective_kind,
    load_trace_payload,
    parse_complete_events,
)
from marin.profiling.xplane import MultipleXPlaneFilesError, find_xplane_file, parse_xplane_timeline

logger = logging.getLogger(__name__)

TOP_KERNELS = 20
_SESSION_STEPS = re.compile(r"steps-(\d+)-to-(\d+)")
# The collective types reported separately; anything else counts as "other".
COLLECTIVES = ("all-to-all", "all-gather", "reduce-scatter", "all-reduce")
_COMM_NAME = re.compile(
    r"nccl|all-to-all|alltoall|all_to_all|all-gather|allgather|all_gather|reduce-scatter|reducescatter|"
    r"all-reduce|allreduce|collective-permute|sendrecv|send-recv",
    re.IGNORECASE,
)
_D2H = re.compile(r"d2h|devicetohost|device_to_host|memcpydtoh", re.IGNORECASE)
_H2D = re.compile(r"h2d|hosttodevice|host_to_device|memcpyhtod", re.IGNORECASE)
# NCCL implements all-to-all as grouped send/recv kernels; XLA's op name on the event says which collective it is.
_NCCL_KIND = (
    ("alltoall", "all-to-all"),
    ("allgather", "all-gather"),
    ("reducescatter", "reduce-scatter"),
    ("allreduce", "all-reduce"),
)


@dataclasses.dataclass(frozen=True)
class Kernel:
    name: str
    seconds_per_step: float
    share_of_step: float


@dataclasses.dataclass(frozen=True)
class Exposure:
    seconds_per_step: float
    exposed_seconds_per_step: float


@dataclasses.dataclass(frozen=True)
class StepProfile:
    """Per-step averages over the profiled steps and devices (seconds)."""

    steps: int
    devices: int
    step_time: float
    compute: float
    communication: float
    collectives: dict[str, Exposure]
    offload: dict[str, Exposure]
    top_kernels: list[Kernel]
    kernel_seconds: dict[str, float]
    """Every kernel's seconds per step and device, for diffs."""
    peak_memory_bytes: int | None = None

    @property
    def exposed_communication(self) -> float:
        return sum(e.exposed_seconds_per_step for e in self.collectives.values())


def kernel_kind(event: TraceEvent) -> str:
    """``compute``, ``d2h``, ``h2d`` or a collective type for one device-stream event."""
    names = " ".join(n for n in (event.name, event.long_name or "", event.tf_op or "") if n)
    if _D2H.search(event.name):
        return "d2h"
    if _H2D.search(event.name):
        return "h2d"
    if not _COMM_NAME.search(names):
        return "compute"
    lowered = names.lower().replace("-", "").replace("_", "")
    for token, kind in _NCCL_KIND:
        if token in lowered:
            return kind
    kind = collective_kind(names)
    return kind if kind in COLLECTIVES else "other-collective"


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _exposed(interval: tuple[float, float], covered: list[tuple[float, float]], starts: list[float]) -> float:
    """The part of ``interval`` not covered by the merged, sorted ``covered`` intervals."""
    start, end = interval
    hidden = 0.0
    i = max(0, bisect.bisect_right(starts, start) - 1)
    while i < len(covered) and covered[i][0] < end:
        lo, hi = max(start, covered[i][0]), min(end, covered[i][1])
        if hi > lo:
            hidden += hi - lo
        i += 1
    return (end - start) - hidden


def step_profile(device_events: Iterable[TraceEvent], *, steps: int, step_time: float) -> StepProfile:
    """Summarize device-stream (kernel) events of ``steps`` profiled steps; times in the trace's microseconds."""
    per_device: dict[str, list[TraceEvent]] = defaultdict(list)
    for event in device_events:
        if event.dur > 0:
            per_device[event.process_name or str(event.pid)].append(event)
    if not per_device or steps <= 0:
        raise ValueError("no device kernel events or no profiled steps")
    devices = len(per_device)
    scale = 1e-6 / (steps * devices)  # microseconds summed over devices and steps -> seconds per step per device

    totals: dict[str, float] = defaultdict(float)
    exposed: dict[str, float] = defaultdict(float)
    kernels: dict[str, float] = defaultdict(float)
    for events in per_device.values():
        kinds = [kernel_kind(e) for e in events]
        compute = _union([(e.ts, e.ts + e.dur) for e, k in zip(events, kinds, strict=True) if k == "compute"])
        starts = [c[0] for c in compute]
        for event, kind in zip(events, kinds, strict=True):
            totals[kind] += event.dur
            kernels[event.name] += event.dur
            if kind != "compute":
                exposed[kind] += _exposed((event.ts, event.ts + event.dur), compute, starts)

    collective_kinds = [k for k in totals if k not in ("compute", "d2h", "h2d")]
    top = sorted(kernels.items(), key=lambda kv: -kv[1])[:TOP_KERNELS]
    return StepProfile(
        steps=steps,
        devices=devices,
        step_time=step_time,
        compute=totals["compute"] * scale,
        communication=sum(totals[k] for k in collective_kinds) * scale,
        collectives={k: Exposure(totals[k] * scale, exposed[k] * scale) for k in sorted(collective_kinds)},
        offload={k: Exposure(totals[k] * scale, exposed[k] * scale) for k in ("d2h", "h2d") if k in totals},
        top_kernels=[Kernel(name, t * scale, t * scale / step_time if step_time else 0.0) for name, t in top],
        kernel_seconds={name: t * scale for name, t in kernels.items()},
    )


def _stream_tracks(profile_dir: Path) -> tuple[Iterator[TraceEvent], list[TraceEvent]]:
    """Device kernel events (GPU stream lines) and step markers of one profile session."""
    try:
        # Every device line in detail: the XLA Modules line times the steps when there is no Steps line.
        timeline = parse_xplane_timeline(find_xplane_file(profile_dir), breakdown_mode=BreakdownMode.EXCLUSIVE_GLOBAL)
        all_tracks = list(timeline.tracks)
        tracks = [t for t in all_tracks if isinstance(t, TraceEventTrack)]
        # The Steps line is summarized as an aggregate, which keeps its step markers.
        aggregated_steps = [e for t in all_tracks if isinstance(t, TraceTrackAggregate) for e in t.step_events]
    except (FileNotFoundError, MultipleXPlaneFilesError):
        aggregated_steps = []
        events, process_names, thread_names = parse_complete_events(
            load_trace_payload(find_profile_trace(profile_dir)).traceEvents
        )
        for event in events:
            event.process_name = process_names.get(event.pid)
            event.thread_name = thread_names.get((event.pid, event.tid))
        tracks = [TraceEventTrack(-1, -1, None, None, events)]
    kernels: list[TraceEvent] = []
    steps: list[TraceEvent] = [e for e in aggregated_steps if e.thread_name == "Steps"]
    for track in tracks:
        for event in track.events:
            if track.process_name is not None:
                event.process_name, event.thread_name = track.process_name, track.thread_name
            device = (event.process_name or "").startswith("/device:")
            if device and (event.thread_name or "").startswith("Stream #"):
                kernels.append(event)
            elif device and event.thread_name in ("Steps", "XLA Modules"):
                steps.append(event)
    return iter(kernels), _step_markers(steps)


def _step_markers(events: list[TraceEvent]) -> list[TraceEvent]:
    """The device step markers: the Steps line when there is one, else the executions of the program that takes the
    most device time on the XLA Modules line (the train step)."""
    marked = [e for e in events if e.thread_name == "Steps"]
    if marked:
        return marked
    modules = [e for e in events if e.thread_name == "XLA Modules"]
    totals: dict[str, float] = defaultdict(float)
    for event in modules:
        totals[event.name] += event.dur
    if not totals:
        return []
    train_step = max(totals, key=totals.__getitem__)
    return [e for e in modules if e.name == train_step]


def profile_dir_summary(profile_dir: Path) -> StepProfile:
    """The summary of one downloaded profile session directory. Step time comes from the devices' step markers, or,
    when the trace has none (JAX's GPU traces have no Steps line), from the span of the device kernels over the steps
    the session's directory name records (``steps-<first>-to-<end>``)."""
    kernels, markers = _stream_tracks(profile_dir)
    kernels = list(kernels)
    by_device: dict[str, list[float]] = defaultdict(list)
    for event in markers:
        if (event.process_name or "").startswith("/device:"):
            by_device[event.process_name or ""].append(event.dur)
    durations = next(iter(by_device.values()), [])
    if durations:
        return step_profile(kernels, steps=len(durations), step_time=sorted(durations)[len(durations) // 2] * 1e-6)
    named = _SESSION_STEPS.search(profile_dir.name)
    if named is None or not kernels:
        raise ValueError(f"{profile_dir}: no device step markers and no steps-<first>-to-<end> session name")
    steps = int(named.group(2)) - int(named.group(1))
    span = max(e.ts + e.dur for e in kernels) - min(e.ts for e in kernels)
    return step_profile(kernels, steps=steps, step_time=span * 1e-6 / steps)


def wandb_metrics(profile: StepProfile) -> dict[str, float]:
    """``profile/*`` keys for the run's W&B summary."""
    metrics = {
        "profile/step_time_s": profile.step_time,
        "profile/compute_s": profile.compute,
        "profile/communication_s": profile.communication,
        "profile/exposed_communication_s": profile.exposed_communication,
    }
    for kind, exposure in {**profile.collectives, **profile.offload}.items():
        metrics[f"profile/{kind}_s"] = exposure.seconds_per_step
        metrics[f"profile/{kind}_exposed_s"] = exposure.exposed_seconds_per_step
    for rank, kernel in enumerate(profile.top_kernels, 1):
        metrics[f"profile/top{rank:02d}_share"] = kernel.share_of_step
    if profile.peak_memory_bytes is not None:
        metrics["profile/peak_memory_gib"] = profile.peak_memory_bytes / 2**30
    return metrics


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:9.1f} ms"


def render(profile: StepProfile, title: str = "profile") -> str:
    """A plain-text summary for logs and terminals."""
    share = (lambda s: f"{100 * s / profile.step_time:5.1f}%") if profile.step_time else (lambda s: "   - ")
    lines = [
        f"{title}: {profile.steps} steps on {profile.devices} device(s); per step and device:",
        f"  step time        {_ms(profile.step_time)}  (device span; includes the profiler's own overhead)",
        f"  compute          {_ms(profile.compute)} {share(profile.compute)}",
        f"  communication    {_ms(profile.communication)} {share(profile.communication)}, exposed "
        f"{_ms(profile.exposed_communication)} {share(profile.exposed_communication)}",
    ]
    for kind, e in profile.collectives.items():
        lines.append(f"    {kind:<16}{_ms(e.seconds_per_step)}, exposed {_ms(e.exposed_seconds_per_step)}")
    for kind, e in profile.offload.items():
        lines.append(f"  offload {kind:<8} {_ms(e.seconds_per_step)}, exposed {_ms(e.exposed_seconds_per_step)}")
    if profile.peak_memory_bytes is not None:
        lines.append(f"  peak memory      {profile.peak_memory_bytes / 2**30:9.1f} GiB")
    lines.append(f"  top {len(profile.top_kernels)} kernels:")
    for k in profile.top_kernels:
        lines.append(f"    {_ms(k.seconds_per_step)} {100 * k.share_of_step:5.1f}%  {k.name[:110]}")
    return "\n".join(lines)


def render_diff(before: StepProfile, after: StepProfile, top: int = 25) -> str:
    """Per-kernel time change (after - before, per step and device) over every kernel, largest first."""
    a, b = before.kernel_seconds, after.kernel_seconds
    rows = sorted(((b.get(n, 0.0) - a.get(n, 0.0), n) for n in a.keys() | b.keys()), key=lambda r: -abs(r[0]))
    lines = [
        f"step time {_ms(before.step_time)} -> {_ms(after.step_time)}",
        f"exposed communication {_ms(before.exposed_communication)} -> {_ms(after.exposed_communication)}",
        f"per-kernel change, the {top} largest:",
    ]
    lines += [f"  {delta * 1e3:+9.2f} ms  {name[:110]}" for delta, name in rows[:top]]
    return "\n".join(lines)


def run_sessions(run_id: str, xprof_root: str | None = None) -> list[str]:
    """The run's uploaded profile session directories, under ``xprof_root`` (default: the XProf root this process
    would upload to, which is the run's when it runs in the run's region)."""
    run_root = (
        StoragePath(f"{xprof_root}/{run_id}")
        if xprof_root
        else StoragePath(XprofUploadConfig().destination_for_run(run_id))
    )
    root = run_root / "plugins" / "profile"
    return sorted(str(p) for p in root.ls()) if root.exists() else []


def download_session(session: str, into: Path) -> Path:
    local = into / Path(session.rstrip("/")).name
    StoragePath(session).download_to(str(local), recursive=True)
    return local


def summarize_run(run_id: str, workdir: Path, xprof_root: str | None = None) -> StepProfile:
    sessions = run_sessions(run_id, xprof_root)
    if not sessions:
        raise SystemExit(f"{run_id}: no uploaded profile sessions")
    # The first process's session: one host's devices, as the in-run summary reports.
    session = next((s for s in sessions if s.rstrip("/").endswith("process0")), sessions[0])
    print(f"{run_id}: reading {session}")
    into = workdir / run_id  # one directory per run: two runs' sessions often share a name
    into.mkdir(parents=True, exist_ok=True)
    return profile_dir_summary(download_session(session, into))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize a run's uploaded profile.")
    parser.add_argument("run")
    parser.add_argument("--compare", help="a second run: also print a per-kernel diff against it")
    parser.add_argument("--xprof-root", help="the runs' XProf root, e.g. s3://<bucket>/tmp/ttl=30d/xprof")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        profile = summarize_run(args.run, Path(tmp), args.xprof_root)
        print(render(profile, args.run))
        if args.compare:
            other = summarize_run(args.compare, Path(tmp), args.xprof_root)
            print(render(other, args.compare))
            print(render_diff(other, profile))


if __name__ == "__main__":
    main()
