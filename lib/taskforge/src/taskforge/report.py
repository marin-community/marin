# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import PurePosixPath

from rigging.filesystem.factory import url_to_fs

from taskforge.checkpoint import FORMAT_1, HEAD, read_journal
from taskforge.effects.events import Resolved, ScopeEntered, ScopeExited, StageCompleted, event_digest
from taskforge.effects.issues import UnsupportedFormat
from taskforge.effects.log import Billing, decode
from taskforge.harness.tools.evidence import ListTool, ReadTool, SearchTool
from taskforge.model.stage import Stage
from taskforge.model.workpiece import Terminal
from taskforge.resources.env import RunCommand, WriteFile
from taskforge.resources.inference import InferenceRequest
from taskforge.resources.rollouts import SolverGeneration, SolverRollout

EVENT_GAPS = {RunCommand.kind: "RunCommand", WriteFile.kind: "WriteFile", InferenceRequest.kind: "InferenceRequest"}
"""The request kinds whose wall-clock gaps are reported, with the labels dashboards key on."""


def inferred(request: InferenceRequest, scopes: Mapping[str, Sequence[float]]) -> Billing:
    """The role a row written before format 10 charges its inference to, guessed from the scopes open when it
    resolved: the reviewer by its evidence tools, and the final review as the reviewer inside the confirmation."""
    if {tool.name for tool in request.tools} >= {ListTool.name, ReadTool.name, SearchTool.name}:
        return Billing.FINAL_REVIEW if scopes["confirm"] else Billing.SOLVE_REVIEW
    adversary = any(v for k, v in scopes.items() if k.startswith("adversary:"))
    return Billing.JUDGE if scopes["judge"] else Billing.ADVERSARY if adversary else Billing.AUTHOR


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
    heads: list[PurePosixPath] = []
    legacy: set[PurePosixPath] = set()
    for path in map(PurePosixPath, await asyncio.to_thread(fs.find, prefix)):
        if path.name == HEAD:
            heads.append(path.parent)
        elif path.name in FORMAT_1:
            legacy.add(path.parent)
        elif path.parent.name in FORMAT_1:  # A part under events.jsonl.parts/.
            legacy.add(path.parent.parent)
    named = ", ".join(sorted(map(str, legacy))[:5]) + (f" and {len(legacy) - 5} more" if len(legacy) > 5 else "")
    if not heads:
        found = f"; it holds {len(legacy)} journal format 1 roots ({named}), migrate them first" if legacy else ""
        raise ValueError(f"No workpiece journals at {root}{found}")
    counts, reasons = Counter[str](), Counter[tuple[str, str]]()
    tokens: dict[str, Counter[str]] = defaultdict(Counter)
    durations: dict[str, list[float]] = defaultdict(list)
    unbilled = Counter[str]()
    accepted: list[float] = []
    unsupported: list[str] = []
    gate = asyncio.Semaphore(32)

    async def collect(path: PurePosixPath) -> None:
        async with gate:
            try:
                journal = await asyncio.to_thread(read_journal, fs.unstrip_protocol(str(path)))
            except UnsupportedFormat as error:  # Named below; a migration brings it to this format.
                unsupported.append(str(error))
                return
            if journal is None:
                return
            status = journal.head.status
            records = await asyncio.to_thread(decode, await asyncio.to_thread(journal.events))
            # A running root is current from its latest claim or checkpoint, even when its newest event is old.
            latest = records[-1].wall_time if records else -math.inf
            if max(latest, -math.inf if status.terminal else journal.head.published_at) < cutoff:
                return
            terminal = str(status.terminal or "running")
            counts[terminal] += 1
            if status.terminal in (Terminal.HELD, Terminal.ABANDONED):
                workpiece = await asyncio.to_thread(journal.workpiece)
                last = workpiece.decisions[-1] if workpiece is not None and workpiece.decisions else None
                for reason in {reason_class(value) for value in getattr(last, "reasons", ())} or {"unknown"}:
                    reasons[terminal, reason] += 1
            generated: set[tuple[str, object]] = set()
            decision_time = None
            scopes: dict[str, list[float]] = defaultdict(list)
            previous = records[0].wall_time if records else 0.0
            for record in records:
                event, stamp = record.event, record.wall_time
                if isinstance(event, StageCompleted) and event.terminal is Terminal.ACCEPTED:
                    decision_time = stamp
                # A checkpointed generation is charged by its own row, and the slot results after it are graded from
                # it; a slot result before it paid for an earlier generation. Tracked ahead of the window.
                if isinstance(event, Resolved) and isinstance(event.request, SolverGeneration):
                    generated.add((event_digest(event.request.slot), event.fingerprints))
                if isinstance(event, ScopeEntered):
                    if event.name == "workpiece":
                        scopes.clear()
                    scopes[event.name].append(stamp)
                elif isinstance(event, ScopeExited) and scopes[event.name]:
                    start = scopes[event.name].pop()
                    if event.name in Stage and stamp >= cutoff:
                        durations[f"stage={event.name}"].append(max(0, stamp - start))
                elif isinstance(event, Resolved) and not event.replayed and stamp >= cutoff:
                    if label := EVENT_GAPS.get(event.request.kind):
                        durations[f"event_gap={label}"].append(max(0, stamp - previous))
                    # A row from before format 10 names no payer: its role is guessed from the open scopes.
                    if isinstance(event.request, InferenceRequest):
                        unbilled["calls"] += record.billed is None
                        tokens[record.billed or inferred(event.request, scopes)].update(asdict(event.value.usage))
                    elif isinstance(event.request, SolverGeneration) or (
                        isinstance(event.request, SolverRollout)
                        and (event_digest(event.request), event.fingerprints) not in generated
                    ):
                        unbilled["calls"] += record.billed is None
                        rollout = event.value if isinstance(event.request, SolverGeneration) else event.value.rollout
                        role = record.billed or (Billing.CONFIRMATION if scopes["confirm"] else Billing.SOLVER)
                        for conversation in rollout.conversations:
                            for step in conversation.steps:
                                tokens[role].update(
                                    step.turn.metadata.get("usage", asdict(step.turn.usage) if step.turn.usage else {})
                                )
                previous = stamp
            if status.terminal is Terminal.ACCEPTED and decision_time is not None:
                accepted.append(decision_time)

    await asyncio.gather(*(collect(path) for path in sorted(heads)))
    lines = [f"workpieces={sum(counts.values())} since_hours={since_hours if since_hours is not None else 'all'}"]
    if legacy:
        lines.append(f"skipped journal_format=1 count={len(legacy)} roots={named}")
    lines += [f"skipped unsupported_format {error}" for error in sorted(unsupported)]
    lines += [f"terminal={terminal} count={counts[terminal]}" for terminal in ("running", *Terminal)]
    lines += [
        f"terminal={terminal} reason={reason} count={count}" for (terminal, reason), count in sorted(reasons.items())
    ]
    for hours in (1, 6, 24):
        count = sum(max(cutoff, now - hours * 3600) <= stamp <= now for stamp in accepted)
        rate = count / min(hours, since_hours or hours)
        lines.append(f"accepted_per_hour window_hours={hours} count={count} rate={rate:.3f}")
    for role in Billing:
        usage = " ".join(f"{name}={tokens[role][name + '_tokens']}" for name in ("prompt", "cached", "completion"))
        lines.append(f"tokens role={role} {usage}")
    if unbilled["calls"]:
        lines.append(f"tokens attribution=inferred calls={unbilled['calls']}")
    for key in (
        *[f"stage={stage}" for stage in Stage],
        *[f"event_gap={label}" for label in EVENT_GAPS.values()],
    ):
        values = sorted(durations[key])
        quantiles = [f"{values[math.ceil(len(values) * q) - 1]:.3f}" if values else "n/a" for q in (0.5, 0.9)]
        lines.append(f"{key} n={len(values)} p50_seconds={quantiles[0]} p90_seconds={quantiles[1]}")
    return tuple(lines)
