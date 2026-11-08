# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict

from pydantic import TypeAdapter
from rigging.filesystem.factory import open_url, url_to_fs

from forge.effects.events import Resolved, ScopeEntered, ScopeExited, StageCompleted
from forge.effects.log import JsonlSink, Status
from forge.model.stage import Stage
from forge.model.workpiece import WORKPIECE, Terminal, Workpiece
from forge.resources.inference import InferenceRequest
from forge.resources.rollouts import SolverRollout


def artifacts(path: str) -> tuple[Status, Workpiece]:
    with open_url(path + "/status.json", "rb") as source:
        status = TypeAdapter(Status).validate_json(source.read())
    with open_url(path + "/workpiece.json", "rb") as source:
        return status, WORKPIECE.validate_json(source.read())


def reason_class(reason: str) -> str:
    if reason.startswith("bug:"):
        pattern = r"(?m)^\s*(?:[|+]\s*)*(?:\w+\.)*([A-Z]\w*)(?=:|$)"
        types = re.findall(pattern, reason.removeprefix("bug:").strip())
        types = [name for name in types if name not in {"Traceback", "ExceptionGroup", "BaseExceptionGroup"}]
        return "bug:" + (types[0] if types else "unknown")
    return reason.split(":", 1)[0].split(" <- ", 1)[0].strip()


async def report(root: str, since_hours: float | None = None) -> tuple[str, ...]:
    if since_hours is not None and (not math.isfinite(since_hours) or since_hours <= 0):
        raise ValueError("since-hours must be finite and positive")
    now = time.time()
    cutoff = now - since_hours * 3600 if since_hours is not None else -math.inf
    fs, prefix = url_to_fs(root.rstrip("/"))
    paths = await asyncio.to_thread(fs.glob, prefix + "/**/status.json")
    if not paths:
        raise ValueError(f"No workpiece status files at {root}")
    counts, reasons = Counter[str](), Counter[tuple[str, str]]()
    tokens: dict[str, Counter[str]] = defaultdict(Counter)
    durations: dict[str, list[float]] = defaultdict(list)
    accepted: list[float] = []
    gate = asyncio.Semaphore(32)

    async def collect(path: str) -> None:
        async with gate:
            directory = fs.unstrip_protocol(path).removesuffix("/status.json")
            (status, workpiece), records = await asyncio.gather(
                asyncio.to_thread(artifacts, directory), JsonlSink(directory + "/events.jsonl").read()
            )
            if not records or records[-1].wall_time < cutoff:
                return
            terminal = str(status.terminal or "running")
            counts[terminal] += 1
            if status.terminal in (Terminal.HELD, Terminal.ABANDONED):
                last = workpiece.decisions[-1] if workpiece.decisions else None
                for reason in {reason_class(value) for value in getattr(last, "reasons", ())} or {"unknown"}:
                    reasons[terminal, reason] += 1
            decision_time = None
            scopes: dict[str, list[float]] = defaultdict(list)
            previous = records[0].wall_time
            for record in records:
                event, stamp = record.event, record.wall_time
                if isinstance(event, StageCompleted) and event.terminal is Terminal.ACCEPTED:
                    decision_time = stamp
                if isinstance(event, ScopeEntered):
                    if event.name == "workpiece":
                        scopes.clear()
                    scopes[event.name].append(stamp)
                elif isinstance(event, ScopeExited) and scopes[event.name]:
                    start = scopes[event.name].pop()
                    if event.name in Stage and stamp >= cutoff:
                        durations[f"stage={event.name}"].append(max(0, stamp - start))
                elif isinstance(event, Resolved) and not event.replayed and stamp >= cutoff:
                    name = type(event.request).__name__
                    if name in {"RunCommand", "WriteFile", "InferenceRequest"}:
                        durations[f"event_gap={name}"].append(max(0, stamp - previous))
                    if isinstance(event.request, InferenceRequest):
                        adversary = any(v for k, v in scopes.items() if k.startswith("adversary:"))
                        role = "judge" if scopes["judge"] else "adversary" if adversary else "author"
                        tokens[role].update(asdict(event.value.usage))
                    elif isinstance(event.request, SolverRollout):
                        for conversation in event.value.rollout.conversations:
                            for step in conversation.steps:
                                tokens["solver"].update(
                                    step.turn.metadata.get("usage", asdict(step.turn.usage) if step.turn.usage else {})
                                )
                previous = stamp
            if status.terminal is Terminal.ACCEPTED and decision_time is not None:
                accepted.append(decision_time)

    await asyncio.gather(*(collect(path) for path in sorted(paths)))
    lines = [f"workpieces={sum(counts.values())} since_hours={since_hours if since_hours is not None else 'all'}"]
    lines += [f"terminal={terminal} count={counts[terminal]}" for terminal in ("running", *Terminal)]
    lines += [
        f"terminal={terminal} reason={reason} count={count}" for (terminal, reason), count in sorted(reasons.items())
    ]
    for hours in (1, 6, 24):
        count = sum(max(cutoff, now - hours * 3600) <= stamp <= now for stamp in accepted)
        rate = count / min(hours, since_hours or hours)
        lines.append(f"accepted_per_hour window_hours={hours} count={count} rate={rate:.3f}")
    for role in ("author", "solver", "adversary", "judge"):
        usage = " ".join(f"{name}={tokens[role][name + '_tokens']}" for name in ("prompt", "cached", "completion"))
        lines.append(f"tokens role={role} {usage}")
    for key in (
        *[f"stage={stage}" for stage in Stage],
        *[f"event_gap={name}" for name in ("RunCommand", "WriteFile", "InferenceRequest")],
    ):
        values = sorted(durations[key])
        quantiles = [f"{values[math.ceil(len(values) * q) - 1]:.3f}" if values else "n/a" for q in (0.5, 0.9)]
        lines.append(f"{key} n={len(values)} p50_seconds={quantiles[0]} p90_seconds={quantiles[1]}")
    return tuple(lines)
