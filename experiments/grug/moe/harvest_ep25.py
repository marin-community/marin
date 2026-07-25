#!/usr/bin/env python3
"""Harvest step-indexed drop/MFU/loss series for an EP25 grug job via finelog SQL.

Usage: uv run python experiments/grug/moe/harvest_ep25.py <job-name> [--baseline <job-name>]

Reads the json_logger metric lines from the log namespace (works while the
fetch_logs batch path lags) and prints: per-step drop_fraction checkpoints,
tail-100 mean, p50/p10/p90 MFU, and loss tail20.
"""

import json
import sys

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient

CHECKPOINTS = (0, 5, 30, 60, 90, 119, 150, 200, 250, 275, 300, 325, 349)


def fetch_metrics(job_name: str) -> list[dict]:
    with connect_controller(cluster_name="marin") as ep:
        client = IrisClient.remote(ep.url, credentials=ep.credentials)
        logc = client._cluster_client._log_client
        t = logc.query(
            "select data from log where key like %s and data like %s order by key, epoch_ms" % tuple(
                "'{}'".format(s.replace("'", "''"))
                for s in (f"/mwittmann/{job_name}/grug-train-%/0:0", '%"tracker": "json_logger"%')
            ),
            max_rows=100_000,
        )
    rows = []
    for r in t.to_pylist():
        i = r["data"].find('{"tracker"')
        if i < 0:
            continue
        try:
            rows.append(json.loads(r["data"][i:]))
        except json.JSONDecodeError:
            continue
    return rows


def summarize(job_name: str) -> None:
    rows = fetch_metrics(job_name)
    by_step: dict[int, dict] = {}
    last_summary: dict = {}
    for r in rows:
        if r.get("event") == "summary":
            last_summary = r["metrics"]
            continue
        if "step" not in r:
            continue
        step = int(r["step"])
        # Multiple log rows share a step (loss/drops, throughput, optim) — merge, don't overwrite.
        by_step.setdefault(step, {}).update(r["metrics"])
    if not by_step:
        print(f"{job_name}: NO METRIC ROWS (ingestion lag?)")
        return
    steps = sorted(by_step)
    drops = {s: by_step[s].get("moe/drop_fraction") for s in steps}
    losses = {s: by_step[s].get("train/loss") for s in steps}
    print(f"== {job_name}: {len(steps)} steps, range {steps[0]}..{steps[-1]}")
    print("drop_fraction:", " ".join(f"{drops[s]:.3f}({s})" for s in CHECKPOINTS if s in drops and drops[s] is not None))
    tail = [drops[s] for s in steps if s >= steps[-1] - 99 and drops[s] is not None]
    if tail:
        print(f"drop tail-{len(tail)} mean {sum(tail) / len(tail):.4f}, last-10 mean {sum(tail[-10:]) / 10:.4f}")
    loss_tail = [losses[s] for s in steps[-20:] if losses[s] is not None]
    if loss_tail:
        print(f"loss last {losses[steps[-1]]:.4f} tail20 {sum(loss_tail) / len(loss_tail):.4f}")
    if last_summary:
        tp = {k.replace("throughput/", ""): round(v, 3) for k, v in last_summary.items() if k.startswith("throughput/")}
        print("summary:", json.dumps(tp, sort_keys=True))


if __name__ == "__main__":
    for name in sys.argv[1:]:
        summarize(name)
