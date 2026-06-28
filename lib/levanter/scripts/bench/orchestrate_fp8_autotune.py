# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU orchestrator for the FP8 ragged-dot autotuner: shards the sweep across GPUs as subprocesses.

The single-GPU harness (``bench_ragged_fp8_autotune.py``) injects a candidate block config by mutating a
process-global module attribute and busts the jit cache with ``jax.clear_caches()`` — both process-wide,
so candidates cannot run concurrently *within* one process. This orchestrator instead runs one worker
*subprocess per GPU* (pinned via ``CUDA_VISIBLE_DEVICES``); each worker evaluates its slice exactly as
the harness does today (single-GPU, sequential, correct), and the parent owns the coordinate descent.

It stays jax-free (imports only the pure config/stat modules) so the parent never initializes a backend
and never competes with its children for GPU memory.

Waves (barriers between them carry the coordinate-descent dependencies):
  1. bf16 sweep + fp8 Mosaic stage-A sweep + fp8 numerics gate   -> best bf16, best Mosaic, gate pass
  2. fp8 wgrad stage-B sweep at each shape's best Mosaic           -> best wgrad
  3. headline re-time of each shape's fp8 winner and bf16 winner   -> ratio-of-medians speedup + CI
Within a wave, every shape's candidates fan out across all GPUs. The headline pins each shape's fp8 and
bf16 arms to the *same* GPU so clock/thermal cancels in the ratio.

    # on an H100x32 cluster job (after the forked-jaxlib setup)
    uv run python lib/levanter/scripts/bench/orchestrate_fp8_autotune.py --shapes all --out-dir scratch/fp8_sweep

    # local plumbing check on CPU (bf16-only, 2 simulated workers)
    uv run --no-sync python lib/levanter/scripts/bench/orchestrate_fp8_autotune.py --simulate --out-dir /tmp/sim
"""

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_autotune_configs import (  # noqa: E402
    SHAPE_GRID,
    bf16_candidate_dicts,
    mosaic_candidate_dicts,
    wgrad_candidate_dicts,
)
from fp8_autotune_stats import ratio_median_ci, summarize_times  # noqa: E402

_HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_ragged_fp8_autotune.py")


def detect_num_gpus(arg):
    if arg:
        return arg
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        n = len([line for line in out.splitlines() if line.strip().startswith("GPU ")])
        return n or 1
    except Exception:
        return 1


def _chunk(items, n):
    """Split ``items`` into ``n`` contiguous, near-equal chunks (empty chunks dropped)."""
    n = max(1, min(n, len(items)))
    size = math.ceil(len(items) / n)
    return [items[i : i + size] for i in range(0, len(items), size)]


def plan_units(reqs_by_shape, num_gpus, max_reqs_per_worker):
    """Pack each shape's requests into single-shape work units, ~proportional to its share of GPUs.

    Single-shape so a worker builds one shape's inputs/dots once and reuses them. Aim for one batch
    (sum of units <= num_gpus) when possible, while bounding any worker to ``max_reqs_per_worker``.
    """
    total = sum(len(v) for v in reqs_by_shape.values())
    units, uid = [], 0
    for shape_name, reqs in reqs_by_shape.items():
        if not reqs:
            continue
        share = max(1, round(num_gpus * len(reqs) / total)) if total else 1
        workers = max(share, math.ceil(len(reqs) / max_reqs_per_worker))
        for chunk in _chunk(reqs, workers):
            units.append({"uid": uid, "shape": shape_name, "requests": chunk})
            uid += 1
    return units


def _write_spec(unit, common, path):
    shape = SHAPE_GRID[unit["shape"]]
    spec = {"shape": dataclasses.asdict(shape), **common, "requests": unit["requests"]}
    with open(path, "w") as f:
        json.dump(spec, f)


def run_pool(units, common, *, num_gpus, out_dir, tag, simulate, log):
    """Run work units across ``num_gpus`` pinned worker subprocesses; return all parsed result rows."""
    if not units:
        return []
    os.makedirs(out_dir, exist_ok=True)
    queue = list(units)
    free = list(range(num_gpus))
    running = {}  # gpu_id -> (proc, rows_out, unit, logfile)
    rows = []

    def launch(unit, gpu):
        wf = os.path.join(out_dir, f"{tag}_u{unit['uid']}_spec.json")
        ro = os.path.join(out_dir, f"{tag}_u{unit['uid']}_rows.jsonl")
        lf = open(os.path.join(out_dir, f"{tag}_u{unit['uid']}.log"), "w")
        _write_spec(unit, common, wf)
        env = os.environ.copy()
        if simulate:
            env["JAX_PLATFORMS"] = "cpu"
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [sys.executable, _HARNESS, "--worker", "--work-file", wf, "--rows-out", ro]
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        running[gpu] = (proc, ro, unit, lf)

    while queue and free:
        launch(queue.pop(), free.pop())
    log(f"  [{tag}] {len(units)} units across {num_gpus} gpus ({len(running)} running)")

    while running:
        time.sleep(1.0)
        for gpu, (proc, ro, unit, lf) in list(running.items()):
            if proc.poll() is None:
                continue
            lf.close()
            del running[gpu]
            if proc.returncode == 0 and os.path.exists(ro):
                with open(ro) as f:
                    rows.extend(json.loads(line) for line in f if line.strip())
            else:
                log(f"  [{tag}] WORKER u{unit['uid']} (shape={unit['shape']}) failed rc={proc.returncode}; see log")
            if queue:
                launch(queue.pop(), gpu)
            else:
                free.append(gpu)
    return rows


def _best(rows, pred):
    """Lowest-median row among those matching ``pred`` that compiled (no error)."""
    cand = [r for r in rows if pred(r) and not r.get("error") and r.get("steady_state_time_s")]
    return min(cand, key=lambda r: r["steady_state_time_s"]) if cand else None


def _shape_rows(rows, shape_name):
    return [r for r in rows if str(r.get("request_id", "")).startswith(shape_name + "|")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default="all", help="comma list of {small,target,scale} or 'all'")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--grad-dtype", choices=("e4m3", "e5m2"), default="e5m2")
    ap.add_argument("--mosaic-wgrad", choices=("bf16", "fp8"), default="fp8")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--inner-steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--numerics-tol", type=float, default=0.25)
    ap.add_argument("--num-gpus", type=int, default=None, help="default: detect via nvidia-smi")
    ap.add_argument("--max-reqs-per-worker", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--simulate", action="store_true", help="CPU plumbing check: bf16-only, JAX_PLATFORMS=cpu")
    ap.add_argument("--no-fp8", action="store_true", help="bf16 sweep only (skip the mosaic/wgrad waves)")
    args = ap.parse_args()

    num_gpus = 2 if args.simulate else detect_num_gpus(args.num_gpus)
    no_fp8 = args.no_fp8 or args.simulate
    out_dir = args.out_dir or os.path.join("scratch", "fp8_sweep", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    shape_keys = list(SHAPE_GRID) if args.shapes == "all" else [s.strip() for s in args.shapes.split(",")]
    shapes = [SHAPE_GRID[k] for k in shape_keys]
    common = {
        "dtype": args.dtype,
        "grad_dtype": args.grad_dtype,
        "mosaic_wgrad": args.mosaic_wgrad,
        "samples": args.samples,
        "inner_steps": args.inner_steps,
        "warmup": args.warmup,
    }

    def log(msg):
        print(msg, flush=True)

    log(f"orchestrator: num_gpus={num_gpus} shapes={[str(s) for s in shapes]} no_fp8={no_fp8} out_dir={out_dir}")
    default_wgrad = wgrad_candidate_dicts()[0]
    all_rows = []

    # ---- Wave 1: bf16 sweep + fp8 mosaic stage-A (request 0 carries the numerics gate). ----
    reqs1 = {}
    for s in shapes:
        rs = [
            {"id": f"{s.name}|bf16|{i}", "kind": "bf16", "bf16cfg": cfg, "want_times": False}
            for i, cfg in enumerate(bf16_candidate_dicts())
        ]
        if not no_fp8:
            for i, cfg in enumerate(mosaic_candidate_dicts()):
                rs.append(
                    {
                        "id": f"{s.name}|mosaicA|{i}",
                        "kind": "fp8",
                        "mosaic": cfg,
                        "wgrad": default_wgrad,
                        "want_numerics": i == 0,
                        "want_times": False,
                    }
                )
        reqs1[s.name] = rs
    units1 = plan_units(reqs1, num_gpus, args.max_reqs_per_worker)
    rows1 = run_pool(units1, common, num_gpus=num_gpus, out_dir=out_dir, tag="wave1", simulate=args.simulate, log=log)
    all_rows += rows1

    per_shape = {}
    for s in shapes:
        sr = _shape_rows(rows1, s.name)
        info = {"best_bf16": _best(sr, lambda r: r.get("kind") == "bf16")}
        if not no_fp8:
            by_id = {r["request_id"]: r for r in sr}
            gate = by_id.get(f"{s.name}|mosaicA|0")
            gate_rel = gate.get("rel_frob_vs_bf16") if gate else None
            if gate is None or gate.get("error") or gate_rel is None or gate_rel > args.numerics_tol:
                info["fp8_failed"] = f"numerics gate rel_frob={gate_rel} (tol {args.numerics_tol})"
                log(f"  {s.name}: fp8 gate FAILED ({info['fp8_failed']})")
            else:
                info["gate_rel_frob"] = gate_rel
                info["best_mosaic"] = _best(sr, lambda r: str(r.get("request_id")).startswith(f"{s.name}|mosaicA|"))
        per_shape[s.name] = info
        bb = info["best_bf16"]
        log(f"  {s.name}: best bf16 {bb['steady_state_time_s'] * 1e3:.3f} ms" if bb else f"  {s.name}: no bf16")

    # ---- Wave 2: fp8 wgrad stage-B at each shape's best Mosaic (fp8 wgrad mode only). ----
    if not no_fp8 and args.mosaic_wgrad == "fp8":
        reqs2 = {}
        for s in shapes:
            info = per_shape[s.name]
            if "best_mosaic" not in info:
                continue
            bm = info["best_mosaic"]["block_sizes"]["mosaic"]
            reqs2[s.name] = [
                {"id": f"{s.name}|wgradB|{i}", "kind": "fp8", "mosaic": bm, "wgrad": cfg, "want_times": False}
                for i, cfg in enumerate(wgrad_candidate_dicts())
            ]
        units2 = plan_units(reqs2, num_gpus, args.max_reqs_per_worker)
        rows2 = run_pool(units2, common, num_gpus=num_gpus, out_dir=out_dir, tag="wave2", simulate=args.simulate, log=log)
        all_rows += rows2
        for s in shapes:
            if s.name in reqs2:
                best_wg = _best(_shape_rows(rows2, s.name), lambda r: True)
                per_shape[s.name]["best_wgrad"] = best_wg

    # ---- Wave 3: headline re-time of each shape's winners (fp8 + bf16 on the same GPU). ----
    reqs3 = {}
    for s in shapes:
        info = per_shape[s.name]
        rs = []
        if info.get("best_bf16"):
            rs.append({"id": f"{s.name}|hlbf16", "kind": "bf16", "bf16cfg": info["best_bf16"]["block_sizes"], "want_times": True})
        if not no_fp8 and "best_mosaic" in info:
            bm = info["best_mosaic"]["block_sizes"]["mosaic"]
            if args.mosaic_wgrad == "fp8":
                wg = (info.get("best_wgrad") or {}).get("block_sizes", {}).get("wgrad", default_wgrad)
            else:
                wg = default_wgrad
            rs.append(
                {"id": f"{s.name}|hlfp8", "kind": "fp8", "mosaic": bm, "wgrad": wg, "want_times": True, "want_numerics": True}
            )
        reqs3[s.name] = rs
    # One unit per shape so its two arms run back-to-back on one GPU (clean A/B).
    units3 = [{"uid": i, "shape": s.name, "requests": reqs3[s.name]} for i, s in enumerate(shapes) if reqs3[s.name]]
    rows3 = run_pool(units3, common, num_gpus=num_gpus, out_dir=out_dir, tag="wave3", simulate=args.simulate, log=log)
    all_rows += rows3

    results = []
    for s in shapes:
        sr = {r["request_id"]: r for r in _shape_rows(rows3, s.name)}
        entry = {"shape": str(s), "shape_dims": dataclasses.asdict(s)}
        bf = sr.get(f"{s.name}|hlbf16")
        fp = sr.get(f"{s.name}|hlfp8")
        if bf and bf.get("times"):
            entry["bf16_best"] = {"cfg": bf["block_sizes"], **summarize_times(bf["times"])}
        if per_shape[s.name].get("fp8_failed"):
            entry["fp8"] = {"failed": per_shape[s.name]["fp8_failed"]}
        if fp and fp.get("times"):
            entry["fp8_best"] = {
                "mosaic": fp["block_sizes"]["mosaic"],
                "wgrad": fp["block_sizes"]["wgrad"],
                "grad_rel_frob_vs_bf16": fp.get("rel_frob_vs_bf16"),
                **summarize_times(fp["times"]),
            }
        if bf and fp and bf.get("times") and fp.get("times"):
            speedup, lo, hi = ratio_median_ci(fp["times"], bf["times"])
            entry["speedup_vs_bf16_best"] = {"median": speedup, "ci95_low": lo, "ci95_high": hi}
            log(f"  >> HEADLINE {s.name}: {speedup:.3f}x [CI {lo:.3f}-{hi:.3f}]")
        results.append(entry)

    rows_path = os.path.join(out_dir, "rows.jsonl")
    with open(rows_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    summary = {"num_gpus": num_gpus, "no_fp8": no_fp8, **common, "numerics_tol": args.numerics_tol, "results": results}
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {len(all_rows)} rows -> {rows_path}")
    log(f"wrote summary -> {summary_path}")
    log("result_json " + json.dumps({"summary_path": summary_path, "results": results}))


if __name__ == "__main__":
    main()
