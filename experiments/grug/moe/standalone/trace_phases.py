# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-phase GPU time attribution from a JAX xplane profile (MXFP8-005).

Aggregates device-stream kernel time from the ``*.xplane.pb`` a
``jax.profiler.trace`` block wrote, buckets it into step phases (expert MLP,
attention, collectives, producers, dense GEMMs, ...), and prints a numeric
breakdown plus a top-kernels table to stdout. Everything is normalized to
milliseconds per GPU per profiled step so arms are directly comparable.

Classification uses the XLA op metadata path (``tf_op``: jax named scopes /
named_call) first and the raw kernel symbol second; the raw table carries both
so results can be re-bucketed offline from the job logs alone.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from marin.profiling.xplane import find_xplane_file, parse_xplane_timeline

# Ordered (bucket, regex over "tf_op|kernel_name", lowercased). First hit wins.
_BUCKET_RULES: tuple[tuple[str, str], ...] = (
    ("collectives", r"nccl|all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute|ragged-all-to-all"),
    # The fused MXFP8 expert-MLP legs (cutlass DSL kernels) and the bf16
    # grouped GEMMs (triton ragged dot) both live under the moe_up_down scope;
    # kernel-symbol patterns keep them apart when scopes are missing.
    ("expert_mlp_fused", r"mxfp8_fused|blockscaledmoegroupedgemm|grouped_gemm|dswiglu|swiglu_quant"),
    ("expert_mlp_bf16_ragged", r"ragged_dot|_triton_ragged|ragged-dot"),
    ("mxfp8_producers", r"dual_quantize|quantize_mxfp8|build_sf|e8m0|mxfp8"),
    ("attention", r"fmha|flash|attention|fa4|_attn|blackwellfmha"),
    ("moe_dispatch_combine", r"moe_up_down|moe_dispatch|expert_mlp|/moe|scatter|permute_by_global_expert|top_k"),
    ("dense_gemm", r"cublas|gemm|matmul|dot_general|triton_gemm"),
    ("memcpy_transpose", r"memcpy|transpose|copy"),
)
_FALLBACK_BUCKET = "other_fusions"


def _classify(tf_op: str | None, name: str) -> str:
    hay = f"{tf_op or ''}|{name}".lower()
    for bucket, pattern in _BUCKET_RULES:
        if re.search(pattern, hay):
            return bucket
    return _FALLBACK_BUCKET


def analyze(profile_dir: Path, *, steps: int, num_gpus: int, top: int = 60) -> None:
    xplane = find_xplane_file(profile_dir)
    timeline = parse_xplane_timeline(xplane)
    for warning in timeline.quality_warnings:
        print(f"TRACE_PHASES warning: {warning}")

    gpu_pids = {pid for pid, pname in timeline.process_names.items() if "gpu" in pname.lower()}
    print(f"TRACE_PHASES planes: {timeline.process_names}")
    line_totals: dict[tuple[str, str], float] = defaultdict(float)
    for ev in timeline.events:
        if ev.pid in gpu_pids:
            line_totals[(ev.process_name, ev.thread_name)] += ev.dur
    for (pname, tname), total in sorted(line_totals.items(), key=lambda kv: -kv[1]):
        print(f"TRACE_PHASES line: {pname} / {tname}: {total / 1e3:.3f} ms total")

    # Device kernel streams only: skip derived/annotation lines so time is not
    # double counted. JAX GPU xplanes put kernels on "Stream #N" lines.
    def _is_stream(thread_name: str) -> bool:
        return "stream" in thread_name.lower()

    events = [ev for ev in timeline.events if ev.pid in gpu_pids and _is_stream(ev.thread_name)]
    if not events:
        # Fall back to everything on GPU planes (older jaxlib naming).
        events = [ev for ev in timeline.events if ev.pid in gpu_pids]
        print("TRACE_PHASES warning: no 'Stream' lines found; using all GPU-plane events")

    denom = steps * num_gpus
    buckets: dict[str, float] = defaultdict(float)
    kernels: dict[str, list] = {}
    for ev in events:
        bucket = _classify(ev.tf_op, ev.name)
        buckets[bucket] += ev.dur
        entry = kernels.setdefault(ev.name, [0.0, 0, ev.tf_op or "", bucket])
        entry[0] += ev.dur
        entry[1] += 1

    total = sum(buckets.values())
    print(f"\nTRACE_PHASES phase breakdown ({steps} steps x {num_gpus} GPUs; device-stream time, ms/GPU/step):")
    for bucket, dur in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {bucket:26s} {dur / 1e3 / denom:9.3f} ms  ({100 * dur / total:5.1f}%)")
    print(f"  {'TOTAL device-busy':26s} {total / 1e3 / denom:9.3f} ms")

    print(f"\nTRACE_PHASES top {top} kernels (total ms/GPU/step | count | bucket | kernel | sample tf_op):")
    ranked = sorted(kernels.items(), key=lambda kv: -kv[1][0])[:top]
    for name, (dur, count, tf_op, bucket) in ranked:
        print(f"  {dur / 1e3 / denom:9.3f} | {count:6d} | {bucket:24s} | {name[:110]} | {tf_op[:120]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--steps", type=int, required=True, help="number of profiled train steps")
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--top", type=int, default=60)
    args = parser.parse_args()
    analyze(args.profile_dir, steps=args.steps, num_gpus=args.num_gpus, top=args.top)


if __name__ == "__main__":
    main()
