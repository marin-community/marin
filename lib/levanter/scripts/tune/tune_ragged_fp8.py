# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Autotune the Triton grouped (ragged) MoE GEMM, bf16 vs per-tensor FP8, on a representative
shape grid (S5 Grug FP8-on-H100). Follows the add-pallas-kernel skill: a bounded block/tile
config space swept over target shape buckets, every (bucket, path, config) benchmarked with
timing AND failures captured, raw results written as a JSON artifact, and a best-config table
per (bucket, path) derived for review.

The shape buckets are the REAL Grug MoE trial model (GRUG_MOE_TRIAL_MODEL: hidden=1024,
intermediate=512, 64 experts, top_k=4, seq 4096) grouped GEMM as one device sees it under the
two plausible expert-parallel layouts -- NOT the oversized hidden=2048/intermediate=5632/8-expert
shape an earlier microbench guessed (that regime is far more compute-heavy per expert than the
real model and is not representative).

Each (bucket, path, config) runs as an isolated subprocess of bench_ragged_fp8.py: configs here
deliberately include ones that OOM shared memory or crash XLA codegen (the f8 backward), and
process isolation keeps one crash from aborting the sweep while still recording its reason.
"""

import argparse
import json
import os
import re
import subprocess
import sys

# Representative grid: one H100's view of the trial-model grouped GEMM. Both layouts carry the
# same per-device token count (global dispatched 524288 / 8 devices = 65536); they differ only in
# how many experts are local, hence per-expert occupancy -- the key driver of compute vs memory
# binding. EP (8 experts/device) -> ~8192 tok/expert; DP (64 experts replicated) -> ~1024.
_HIDDEN = 1024
_INTERMEDIATE = 512
_BUCKETS = [
    {"name": "ep8", "tokens": 65536, "hidden": _HIDDEN, "intermediate": _INTERMEDIATE, "experts": 8},
    {"name": "dp64", "tokens": 65536, "hidden": _HIDDEN, "intermediate": _INTERMEDIATE, "experts": 64},
]

# Bounded block/tile candidate set: (block_m, block_n, block_k, num_warps, num_stages).
# Covers block_n in {64,128,256} (Hopper f8 wgmma favors wide N), block_k in {32,64,128},
# block_m in {64,128}, and a couple of warp/stage variants. Large block_k is paired with fewer
# stages to fit the ~232KB H100 smem budget.
_CONFIGS = [
    (128, 128, 32, 4, 4),  # current production default
    (128, 128, 64, 4, 4),
    (128, 256, 32, 8, 4),
    (128, 256, 64, 8, 3),
    (64, 128, 64, 4, 4),
    (64, 256, 64, 4, 3),
    (128, 64, 128, 4, 3),
    (64, 256, 128, 8, 2),
]

# Paths to evaluate. The f8 backward is known to hit backend walls (mixed-f8 unsupported;
# same-type e4m3 crashes XLA codegen), so fwd+bwd f8 is only a wall re-check at this NEW shape
# (1 config, both compute modes) rather than a full autotune. bf16 fwd+bwd is the production bar.
_AUTOTUNE_PATHS = [
    {"label": "bf16-fwdbwd", "path": "bf16", "forward_only": False, "f8_compute": None},
    {"label": "bf16-fwd", "path": "bf16", "forward_only": True, "f8_compute": None},
    {"label": "fp8-fwd", "path": "fp8", "forward_only": True, "f8_compute": None},  # passthrough e4m3xe4m3
]
_WALLCHECK_PATHS = [
    {"label": "fp8-fwdbwd-passthrough", "path": "fp8", "forward_only": False, "f8_compute": None},
    {"label": "fp8-fwdbwd-e4m3", "path": "fp8", "forward_only": False, "f8_compute": "e4m3"},
]

_BENCH = os.path.join(os.path.dirname(__file__), "..", "bench", "bench_ragged_fp8.py")
_RESULT_RE = re.compile(r"^result_json (.*)$", re.MULTILINE)
_FAIL_PATTERNS = [
    ("mixed_f8_unsupported", "must have the same element type"),
    ("xla_codegen_crash", "bad optional access"),
    ("smem_oom", "Shared memory size limit exceeded"),
    ("promotion", "TypePromotionError"),
]


def _run_one(bucket, path_spec, config, steps, warmup):
    """Run one (bucket, path, config) as an isolated bench subprocess; return a result record."""
    block_m, block_n, block_k, warps, stages = config
    env = dict(os.environ)
    env.update(
        RAGGED_DOT_BLOCK_M=str(block_m),
        RAGGED_DOT_BLOCK_N=str(block_n),
        RAGGED_DOT_BLOCK_K=str(block_k),
        RAGGED_DOT_NUM_WARPS=str(warps),
        RAGGED_DOT_NUM_STAGES=str(stages),
    )
    if path_spec["f8_compute"] is not None:
        env["RAGGED_DOT_F8_COMPUTE"] = path_spec["f8_compute"]

    cmd = [
        sys.executable,
        os.path.abspath(_BENCH),
        "--implementation",
        "triton",
        "--path",
        path_spec["path"],
        "--tokens",
        str(bucket["tokens"]),
        "--hidden",
        str(bucket["hidden"]),
        "--intermediate",
        str(bucket["intermediate"]),
        "--experts",
        str(bucket["experts"]),
        "--steps",
        str(steps),
        "--warmup",
        str(warmup),
        "--no-print-hlo",
    ]
    if path_spec["forward_only"]:
        cmd.append("--forward-only")

    rec = {
        "bucket": bucket["name"],
        "label": path_spec["label"],
        "config": {"block_m": block_m, "block_n": block_n, "block_k": block_k, "warps": warps, "stages": stages},
    }
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    m = _RESULT_RE.search(proc.stdout)
    if m:
        r = json.loads(m.group(1))
        rec.update(
            ok=True,
            tflops=r["achieved_tflops_per_s"],
            steady_ms=r["steady_time_s"] * 1e3,
            compile_s=r["compile_time_s"],
            mfu=r.get("mfu"),
            rel_frob=r.get("rel_frob_vs_bf16"),
            f8_reaches_gemm=r.get("f8_reaches_gemm"),
        )
        return rec

    blob = proc.stdout + "\n" + proc.stderr
    reason = next((name for name, pat in _FAIL_PATTERNS if pat in blob), "unknown")
    rec.update(ok=False, reason=reason, exit_code=proc.returncode)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--artifact", default=None, help="path to write the raw JSON results array")
    args = ap.parse_args()

    print(f"hardware: {[d.device_kind for d in __import__('jax').devices()]}")
    print(f"buckets: {[b['name'] for b in _BUCKETS]}  configs: {len(_CONFIGS)}")

    records = []
    for bucket in _BUCKETS:
        for path_spec in _AUTOTUNE_PATHS:
            for config in _CONFIGS:
                rec = _run_one(bucket, path_spec, config, args.steps, args.warmup)
                records.append(rec)
                tag = f"{rec['tflops']:7.1f} TF/s {rec['steady_ms']:7.3f}ms" if rec["ok"] else f"FAIL[{rec['reason']}]"
                print(
                    f"  {rec['bucket']:5} {rec['label']:12} bm{config[0]} bn{config[1]} bk{config[2]} "
                    f"w{config[3]} s{config[4]}  -> {tag}"
                )
        # f8 fwd+bwd wall re-check at this shape: default config only (walls are config-independent)
        for path_spec in _WALLCHECK_PATHS:
            rec = _run_one(bucket, path_spec, _CONFIGS[0], args.steps, args.warmup)
            records.append(rec)
            tag = f"{rec['tflops']:7.1f} TF/s" if rec["ok"] else f"FAIL[{rec['reason']}]"
            print(f"  {rec['bucket']:5} {rec['label']:22} (default cfg) -> {tag}")

    if args.artifact:
        with open(args.artifact, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\nartifact written: {args.artifact}")

    # Best-config-per-(bucket, path) table + the fp8-vs-bf16 verdict.
    print("\n===== BEST CONFIG PER (bucket, path) =====")
    best = {}
    for rec in records:
        if not rec["ok"]:
            continue
        key = (rec["bucket"], rec["label"])
        if key not in best or rec["tflops"] > best[key]["tflops"]:
            best[key] = rec
    for (bucket, label), rec in sorted(best.items()):
        c = rec["config"]
        print(
            f"  {bucket:5} {label:12} BEST {rec['tflops']:7.1f} TF/s {rec['steady_ms']:7.3f}ms "
            f"(bm{c['block_m']} bn{c['block_n']} bk{c['block_k']} w{c['warps']} s{c['stages']}) "
            f"rel_frob={rec.get('rel_frob')}"
        )

    print("\n===== VERDICT (per bucket) =====")
    for bucket in _BUCKETS:
        n = bucket["name"]
        bf16_fwd = best.get((n, "bf16-fwd"))
        fp8_fwd = best.get((n, "fp8-fwd"))
        bf16_fb = best.get((n, "bf16-fwdbwd"))
        if bf16_fwd and fp8_fwd:
            ratio = fp8_fwd["tflops"] / bf16_fwd["tflops"]
            print(
                f"  {n}: best fp8 fwd {fp8_fwd['tflops']:.1f} vs best bf16 fwd {bf16_fwd['tflops']:.1f} "
                f"= {ratio:.2f}x  (fp8 {'WINS' if ratio > 1 else 'loses'})"
            )
        if bf16_fb:
            print(f"  {n}: best bf16 fwd+bwd (production bar) = {bf16_fb['tflops']:.1f} TF/s")
        for label in ("fp8-fwdbwd-passthrough", "fp8-fwdbwd-e4m3"):
            r = best.get((n, label)) or next((x for x in records if x["bucket"] == n and x["label"] == label), None)
            status = f"{r['tflops']:.1f} TF/s" if (r and r["ok"]) else f"FAIL[{r['reason']}]" if r else "n/a"
            print(f"  {n}: {label} = {status}")


if __name__ == "__main__":
    main()
