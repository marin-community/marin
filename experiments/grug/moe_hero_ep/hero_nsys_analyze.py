# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reduce an Nsight Systems SQLite export to the numbers the MoK head-to-head needs.

`nsys stats` already produces the standard per-kernel and per-API tables. What it does not
produce is the pair of numbers this comparison turns on: summed kernel time versus the
*union* of kernel intervals (their difference is observable concurrency), and the same split
for NCCL versus everything else. Both are computed here, over the whole capture and clipped
to the `XlaModule:#hlo_module=jit_train_step` NVTX ranges when those exist.

Grid/block/register/shared-memory geometry is carried per kernel name because the fused MoK
megakernel reserves SMs for communication, and the resident-CTA count is the only trace-side
handle on that reservation.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from typing import Any

_NS_PER_S = 1e9


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    lo, hi = intervals[0]
    for start, end in intervals[1:]:
        if start > hi:
            total += hi - lo
            lo, hi = start, end
        elif end > hi:
            hi = end
    return total + (hi - lo)


def _clip(intervals: list[tuple[int, int]], windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not windows:
        return intervals
    out = []
    for lo, hi in windows:
        for start, end in intervals:
            if end > lo and start < hi:
                out.append((max(start, lo), min(end, hi)))
    return out


def analyze(path: str) -> dict[str, Any]:
    conn = sqlite3.connect(path)
    tables = _tables(conn)
    result: dict[str, Any] = {"sqlite": path, "tables": sorted(tables)}

    # ---- device geometry -------------------------------------------------------------------
    # smCount is the denominator for MoK's reserved communication CTAs, so read it from the
    # capture rather than assuming a die configuration.
    for table in ("TARGET_INFO_GPU", "TARGET_INFO_CUDA_GPU", "GPU"):
        if table not in tables:
            continue
        try:
            cursor = conn.execute(f"SELECT * FROM {table}")
            columns = [c[0] for c in cursor.description]
            result["gpu_info"] = {"table": table, "rows": [dict(zip(columns, r, strict=True)) for r in cursor]}
        except sqlite3.Error as exc:
            result["gpu_info_error"] = f"{table}: {exc}"
        break

    # ---- step windows from NVTX -------------------------------------------------------------
    windows: list[tuple[int, int]] = []
    if "NVTX_EVENTS" in tables:
        rows = conn.execute(
            "SELECT e.start, e.end, COALESCE(e.text, s.value) FROM NVTX_EVENTS e "
            "LEFT JOIN StringIds s ON s.id = e.textId "
            "WHERE COALESCE(e.text, s.value) LIKE '%jit_train_step%' AND e.end IS NOT NULL"
        ).fetchall()
        seen = {}
        for start, end, text in rows:
            if "XlaModule" in str(text):
                seen[(start, end)] = text
        windows = sorted(seen)
        result["nvtx_train_ranges"] = [{"start_ns": s, "end_ns": e, "dur_s": (e - s) / _NS_PER_S} for s, e in windows]
    result["train_range_count"] = len(windows)

    # ---- kernels ---------------------------------------------------------------------------
    kernels: dict[str, dict[str, Any]] = {}
    intervals: list[tuple[int, int]] = []
    nccl: list[tuple[int, int]] = []
    other: list[tuple[int, int]] = []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        query = (
            "SELECT k.start, k.end, COALESCE(d.value, s.value) AS name, k.gridX, k.gridY, k.gridZ, "
            "k.blockX, k.blockY, k.blockZ, k.registersPerThread, k.staticSharedMemory, "
            "k.dynamicSharedMemory, k.streamId "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "LEFT JOIN StringIds d ON d.id = k.demangledName "
            "LEFT JOIN StringIds s ON s.id = k.shortName"
        )
        for row in conn.execute(query):
            start, end, name = row[0], row[1], row[2] or "<unknown>"
            grid = (row[3], row[4], row[5])
            block = (row[6], row[7], row[8])
            entry = kernels.get(name)
            if entry is None:
                entry = kernels[name] = {
                    "count": 0,
                    "total_ns": 0,
                    "max_ns": 0,
                    "geometry": defaultdict(int),
                    "regs": row[9],
                    "static_shared": row[10],
                    "dynamic_shared": row[11],
                    "streams": defaultdict(int),
                }
            dur = end - start
            entry["count"] += 1
            entry["total_ns"] += dur
            entry["max_ns"] = max(entry["max_ns"], dur)
            entry["geometry"][f"grid{grid}block{block}"] += 1
            entry["streams"][str(row[12])] += 1
            intervals.append((start, end))
            (nccl if name.startswith("ncclDevKernel") else other).append((start, end))

    for name, entry in kernels.items():
        entry["name"] = name
        entry["geometry"] = dict(sorted(entry["geometry"].items(), key=lambda kv: -kv[1])[:6])
        entry["streams"] = dict(entry["streams"])
    result["kernel_count"] = sum(e["count"] for e in kernels.values())
    result["kernels"] = kernels

    def overlap_block(label: str, win: list[tuple[int, int]]) -> dict[str, float]:
        allk = _clip(list(intervals), win)
        n = _clip(list(nccl), win)
        o = _clip(list(other), win)
        summed = sum(e - s for s, e in allk)
        union = _union_ns(allk)
        nu, ou = _union_ns(n), _union_ns(o)
        return {
            "scope": label,
            "summed_s": summed / _NS_PER_S,
            "union_s": union / _NS_PER_S,
            "concurrency_surplus_s": (summed - union) / _NS_PER_S,
            "nccl_summed_s": sum(e - s for s, e in n) / _NS_PER_S,
            "nccl_union_s": nu / _NS_PER_S,
            "other_union_s": ou / _NS_PER_S,
            "nccl_other_overlap_s": (nu + ou - union) / _NS_PER_S,
            "window_s": sum(hi - lo for lo, hi in win) / _NS_PER_S if win else None,
        }

    result["overlap_all"] = overlap_block("capture", [])
    if windows:
        result["overlap_train"] = overlap_block("train_ranges", windows)

    # ---- memory ----------------------------------------------------------------------------
    memory: dict[str, dict[str, float]] = {}
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for kind, count, total, nbytes in conn.execute(
            "SELECT copyKind, COUNT(*), SUM(end-start), SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
        ):
            memory[f"memcpy_kind_{kind}"] = {"count": count, "ns": total or 0, "bytes": nbytes or 0}
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        row = conn.execute("SELECT COUNT(*), SUM(end-start), SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMSET").fetchone()
        memory["memset"] = {"count": row[0], "ns": row[1] or 0, "bytes": row[2] or 0}
    result["memory"] = memory

    # ---- CUDA runtime API ------------------------------------------------------------------
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        api = {}
        for name, count, total in conn.execute(
            "SELECT s.value, COUNT(*), SUM(r.end-r.start) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id = r.nameId GROUP BY s.value ORDER BY SUM(r.end-r.start) DESC LIMIT 60"
        ):
            api[name] = {"count": count, "ns": total or 0}
        result["cuda_api"] = api

    conn.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = analyze(args.sqlite)
    with open(args.output, "w") as handle:
        json.dump(result, handle)
    print(
        f"{args.sqlite}: kernels={result.get('kernel_count')} train_ranges={result.get('train_range_count')} "
        f"surplus={result.get('overlap_all', {}).get('concurrency_surplus_s')}"
    )


if __name__ == "__main__":
    main()
