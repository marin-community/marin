# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import math
import time
from collections import Counter
from collections.abc import Sequence

from pydantic import TypeAdapter
from rigging.filesystem.factory import open_url, url_to_fs

from taskforge.effects.log import JsonlSink, Status
from taskforge.metrics import GAP_EVENTS, RUNNING, Origin, Role, WorkpieceMetrics, workpiece_metrics
from taskforge.model.llm import Usage
from taskforge.model.stage import Stage
from taskforge.model.workpiece import WORKPIECE, Terminal, Workpiece

ALL_ORIGINS = "all"
ACCEPTANCE_WINDOW_HOURS = (1, 6, 24)


def artifacts(path: str) -> tuple[Status, Workpiece]:
    with open_url(path + "/status.json", "rb") as source:
        status = TypeAdapter(Status).validate_json(source.read())
    with open_url(path + "/workpiece.json", "rb") as source:
        return status, WORKPIECE.validate_json(source.read())


def quantiles(values: Sequence[float]) -> str:
    """Nearest-rank p50 and p90 of ``values`` as report fields."""
    ordered = sorted(values)
    p50, p90 = (f"{ordered[math.ceil(len(ordered) * q) - 1]:.3f}" if ordered else "n/a" for q in (0.5, 0.9))
    return f"n={len(ordered)} p50={p50} p90={p90}"


def summary(rows: Sequence[WorkpieceMetrics], now: float, cutoff: float, since_hours: float | None) -> list[str]:
    """Report lines for one group of workpieces."""
    accepted = [row for row in rows if row.terminal == Terminal.ACCEPTED]
    terminals = Counter(row.terminal for row in rows)
    reasons = Counter((row.terminal, reason) for row in rows for reason in row.reasons)
    lines = [f"workpieces count={len(rows)}"]
    lines += [f"terminal={terminal} count={terminals[terminal]}" for terminal in (RUNNING, *Terminal)]
    lines += [
        f"terminal={terminal} reason={reason} count={count}" for (terminal, reason), count in sorted(reasons.items())
    ]
    for hours in ACCEPTANCE_WINDOW_HOURS:
        stamps = [row.accepted_at for row in accepted if row.accepted_at is not None]
        count = sum(max(cutoff, now - hours * 3600) <= stamp <= now for stamp in stamps)
        rate = count / min(hours, since_hours or hours)
        lines.append(f"accepted_per_hour window_hours={hours} count={count} rate={rate:.3f}")
    lines.append(f"accepted_without_author count={sum(row.author_sessions == 0 for row in accepted)}")
    lines.append(f"author_sessions_per_accepted {quantiles([row.author_sessions for row in accepted])}")
    lines.append(f"author_image_fallbacks count={sum(row.author_image_fallbacks for row in rows)}")
    lines.append(f"command_timeouts count={sum(row.command_timeouts for row in rows)}")
    for role in Role:
        usage = sum((row.tokens.get(role, Usage()) for row in rows), Usage())
        fields = {"prompt": usage.prompt_tokens, "cached": usage.cached_tokens, "completion": usage.completion_tokens}
        lines.append(f"tokens role={role} " + " ".join(f"{name}={value}" for name, value in fields.items()))
        if accepted:
            per = " ".join(f"{name}={value / len(accepted):.0f}" for name, value in fields.items())
            lines.append(f"tokens_per_accepted role={role} {per}")
    finished = [row.finished - row.started for row in rows if row.terminal != RUNNING]
    lines.append(f"end_to_end_seconds {quantiles(finished)}")
    for name in ("initial", "final"):
        rates = [getattr(row, f"{name}_solve_rate") for row in accepted]
        lines.append(f"solve_rate_accepted measured={name} {quantiles([r for r in rates if r is not None])}")
    for stage in Stage:
        lines.append(f"stage={stage} {quantiles([v for row in rows for v in row.stage_seconds.get(stage, ())])}")
    for event in GAP_EVENTS:
        name = event.__name__
        lines.append(f"event_gap={name} {quantiles([v for row in rows for v in row.event_gaps.get(name, ())])}")
    return lines


async def report(root: str, since_hours: float | None = None) -> tuple[str, ...]:
    """Summarize the workpieces under ``root``, overall and by origin, each line prefixed ``origin=``."""
    if since_hours is not None and (not math.isfinite(since_hours) or since_hours <= 0):
        raise ValueError("since-hours must be finite and positive")
    now = time.time()
    cutoff = now - since_hours * 3600 if since_hours is not None else -math.inf
    fs, prefix = url_to_fs(root.rstrip("/"))
    paths = await asyncio.to_thread(fs.glob, prefix + "/**/status.json")
    if not paths:
        raise ValueError(f"No workpiece status files at {root}")
    gate = asyncio.Semaphore(32)

    async def collect(path: str) -> WorkpieceMetrics | None:
        async with gate:
            directory = fs.unstrip_protocol(path).removesuffix("/status.json")
            (status, workpiece), records = await asyncio.gather(
                asyncio.to_thread(artifacts, directory), JsonlSink(directory + "/events.jsonl").read()
            )
        if not records or records[-1].wall_time < cutoff:
            return None
        return workpiece_metrics(status, workpiece, records, since=cutoff)

    rows = [row for row in await asyncio.gather(*(collect(path) for path in sorted(paths))) if row is not None]
    lines = [f"workpieces={len(rows)} since_hours={since_hours if since_hours is not None else 'all'}"]
    groups = [(ALL_ORIGINS, rows), *((origin.value, [row for row in rows if row.origin is origin]) for origin in Origin)]
    for name, group in groups:
        if group or name == ALL_ORIGINS:
            lines += [f"origin={name} {line}" for line in summary(group, now, cutoff, since_hours)]
    return tuple(lines)
