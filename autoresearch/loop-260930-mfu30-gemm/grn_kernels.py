"""Per-kernel timing of the fused gated_rms_norm forward at the hero per-GPU shape, over block configs.

Profiles each config and splits device time between the statistics and output kernels, with
achieved HBM bandwidth against the bytes each kernel must move. Also times XLA's reference.

Usage: python grn_kernels.py --configs "t,sd,od,sw,ss,ow,os;..."
"""

import argparse
import glob
import os
import sys

import jax
import jax.numpy as jnp

from levanter.kernels.pallas.gated_rms_norm import GatedRmsNormBlockSizes, gated_rms_norm, gated_rms_norm_reference

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop-260930-mfu30"))
from overlap import load, plane_events  # noqa: E402

T, D, R = 65536, 6144, 128


def profile(fn, args, tag, reps=5):
    out = fn(*args)
    jax.block_until_ready(out)
    d = f"/tmp/grn_prof/{tag}"
    jax.profiler.start_trace(d)
    for _ in range(reps):
        out = fn(*args)
    jax.block_until_ready(out)
    jax.profiler.stop_trace()
    path = sorted(glob.glob(f"{d}/**/*.xplane.pb", recursive=True))[-1]
    planes = {p.name: p for p in load(path).planes}
    by = {}
    for lname, name, s, e, st in plane_events(planes["/device:GPU:0"]):
        if lname.startswith("Stream"):
            by.setdefault(name[:48], 0.0)
            by[name[:48]] += (e - s) * 1e-9 / reps
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    args = ap.parse_args()
    k = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(k[0], (T, D), jnp.bfloat16)
    w = jnp.ones((D,), jnp.bfloat16)
    wd = (0.02 * jax.random.normal(k[1], (D, R))).astype(jnp.bfloat16)
    wu = (0.1 * jax.random.normal(k[2], (R, D))).astype(jnp.bfloat16)
    tensor_gb = T * D * 2 / 1e9
    ref = jax.jit(lambda *a: gated_rms_norm_reference(*a, eps=1e-5))
    by = profile(ref, (x, w, wd, wu), "ref")
    print(f"reference total {sum(by.values()):.3f} ms: " + ", ".join(f"{n}={t:.3f}" for n, t in sorted(by.items(), key=lambda kv: -kv[1])), flush=True)
    for spec in args.configs.split(";"):
        bs = GatedRmsNormBlockSizes(*[int(v) for v in spec.split(",")])
        fn = jax.jit(lambda *a: gated_rms_norm(*a, eps=1e-5, implementation="pallas_gpu", block_sizes=bs))
        try:
            by = profile(fn, (x, w, wd, wu), spec.replace(",", "_"))
        except Exception as e:  # record lowering/compile failures per config
            print(f"{spec}: ERROR {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        stats = sum(t for n, t in by.items() if "stats" in n)
        outk = sum(t for n, t in by.items() if "output" in n)
        other = sum(by.values()) - stats - outk
        print(
            f"{spec:22s} stats {stats:.3f} ms ({tensor_gb / stats:.2f} TB/s of 1 pass)  output {outk:.3f} ms "
            f"({3 * tensor_gb / outk:.2f} TB/s of 3 passes)  other {other:.3f} ms  total {sum(by.values()):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
