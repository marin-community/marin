"""Time hero-shape bf16 GEMMs through XLA's normal cuBLAS path, with NVML clock/power telemetry.

Each config runs twice:
  burst     : after an idle gap, a handful of back-to-back calls (what XLA's autotuner sees);
  sustained : a continuous loop for --sustain seconds; the rate over the last half is reported
              together with the mean SM clock, power, and clock-event reasons in that window.
With --devices N the same GEMM runs concurrently on N local GPUs (node-level power).

Usage: python gemm_bench.py --configs hero_gemms.json --out result.json [--devices 1] [--sustain 6]
       [--only name,name] [--profile]
"""

import argparse
import ctypes
import json
import os
import re
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

PEAK = 2.5e15
NVML_CLOCK_SM = 1

# Candidate fused shapes (projection fusion hypotheses H-C2/H-C3) on top of the in-situ configs.
EXTRA = [
    # Q/K/V fused forward, dgrad, wgrad.
    dict(name="fused_qkv_fwd", batch=1, m=65536, n=9216, k=6144, a="MK", b="KN", c="MN"),
    dict(name="fused_qkv_dgrad", batch=1, m=65536, n=6144, k=9216, a="MK", b="NK", c="MN"),
    dict(name="fused_qkv_wgrad", batch=1, m=6144, n=9216, k=65536, a="KM", b="KN", c="MN"),
    # Shared expert gate+up for one expert, and for both experts.
    dict(name="fused_gateup1_fwd", batch=1, m=65536, n=6144, k=6144, a="MK", b="KN", c="MN"),
    dict(name="fused_gateup2_fwd", batch=1, m=65536, n=12288, k=6144, a="MK", b="KN", c="MN"),
    dict(name="fused_gateup2_dgrad", batch=1, m=65536, n=6144, k=12288, a="MK", b="NK", c="MN"),
    dict(name="fused_gateup2_wgrad", batch=1, m=6144, n=12288, k=65536, a="KM", b="KN", c="MN"),
    # Both shared experts' down projections as one K=6144 GEMM.
    dict(name="fused_down2_fwd", batch=1, m=65536, n=6144, k=6144, a="MK", b="NK", c="MN"),
    # Everything reading mlp_in: latent_down + router + 2 x (gate, up).
    dict(name="fused_mlpin_fwd", batch=1, m=65536, n=15744, k=6144, a="MK", b="KN", c="MN"),
    # Square reference.
    dict(name="ref_8192cube", batch=1, m=8192, n=8192, k=8192, a="MK", b="KN", c="MN"),
]


class Nvml:
    """Minimal NVML reader over ctypes; returns None readings if NVML is unavailable."""

    def __init__(self, n):
        self.ok = False
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
            if self.lib.nvmlInit_v2() != 0:
                return
            self.handles = []
            for i in range(n):
                h = ctypes.c_void_p()
                if self.lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != 0:
                    return
                self.handles.append(h)
            self.ok = True
        except OSError:
            pass

    def sample(self):
        if not self.ok:
            return None
        out = []
        for h in self.handles:
            clk, mw, temp = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
            reasons = ctypes.c_ulonglong()
            self.lib.nvmlDeviceGetClockInfo(h, NVML_CLOCK_SM, ctypes.byref(clk))
            self.lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(mw))
            self.lib.nvmlDeviceGetTemperature(h, 0, ctypes.byref(temp))
            self.lib.nvmlDeviceGetCurrentClocksThrottleReasons(h, ctypes.byref(reasons))
            out.append((clk.value, mw.value / 1000.0, temp.value, reasons.value))
        return out


class Sampler(threading.Thread):
    def __init__(self, nvml, period=0.05):
        super().__init__(daemon=True)
        self.nvml, self.period, self.samples, self.stop = nvml, period, [], False

    def run(self):
        while not self.stop:
            s = self.nvml.sample()
            if s is not None:
                self.samples.append((time.perf_counter(), s))
            time.sleep(self.period)

    def window(self, t0, t1, ndev=None):
        full = [s for t, s in self.samples if t0 <= t <= t1]
        rows = [s[:ndev] if ndev else s for s in full]
        if not rows:
            return None
        per_gpu = np.array([[[g[0], g[1]] for g in s] for s in full], dtype=float).mean(axis=0).round().tolist()
        arr = np.array([[g[0], g[1], g[2]] for s in rows for g in s], dtype=float)
        reasons = 0
        for s in rows:
            for g in s:
                reasons |= g[3]
        return dict(
            sm_clock_mhz=float(arr[:, 0].mean()),
            sm_clock_min=float(arr[:, 0].min()),
            power_w=float(arr[:, 1].mean()),
            power_max=float(arr[:, 1].max()),
            temp_c=float(arr[:, 2].max()),
            clock_event_reasons=hex(reasons),
            n=len(rows),
            per_gpu_clock_power=per_gpu,
        )


def gemm_fn(cfg):
    def f(a, b):
        a_ = a if cfg["a"] == "MK" else jnp.swapaxes(a, -1, -2)
        b_ = b if cfg["b"] == "KN" else jnp.swapaxes(b, -1, -2)
        c = jnp.matmul(a_, b_)
        return c if cfg["c"] == "MN" else jnp.swapaxes(c, -1, -2)

    f.__name__ = "gemm_" + cfg["name"]
    return jax.jit(f)


def pattern(x, data):
    """Shape the operand bit statistics; tensor-core power depends on them."""
    if data == "normal":
        return x
    if data == "zeros":
        return jnp.zeros_like(x)
    if data == "mant2":  # keep 2 of bf16's 7 mantissa bits
        bits = jax.lax.bitcast_convert_type(x, jnp.uint16)
        return jax.lax.bitcast_convert_type(bits & jnp.uint16(0xFFE0), jnp.bfloat16)
    if data == "sparse50":
        return jnp.where(jnp.arange(x.size, dtype=jnp.int32).reshape(x.shape) % 2 == 0, x, jnp.zeros_like(x))
    if data == "smooth":  # rows vary slowly along the contiguous axis, like correlated activations
        return (x + jnp.roll(x, 1, axis=-1) + jnp.roll(x, 2, axis=-1) + jnp.roll(x, 3, axis=-1)) * 0.5
    raise ValueError(data)


def operands(cfg, dev, key, data="normal"):
    bt, m, n, k = cfg["batch"], cfg["m"], cfg["n"], cfg["k"]
    ashape = (m, k) if cfg["a"] == "MK" else (k, m)
    bshape = (k, n) if cfg["b"] == "KN" else (n, k)
    if bt > 1:
        ashape, bshape = (bt, *ashape), (bt, *bshape)
    ka, kb = jax.random.split(key)
    with jax.default_device(dev):
        a = pattern(jax.random.normal(ka, ashape, dtype=jnp.bfloat16), data)
        b = pattern((jax.random.normal(kb, bshape, dtype=jnp.bfloat16) * (1.0 / float(np.sqrt(k)))).astype(jnp.bfloat16), data)
    assert a.dtype == jnp.bfloat16 and b.dtype == jnp.bfloat16, (a.dtype, b.dtype)
    return jax.device_put(a, dev), jax.device_put(b, dev)


def run_config(cfg, devs, sampler, sustain, idle, data="normal"):
    fn = gemm_fn(cfg)
    ops = [operands(cfg, d, jax.random.PRNGKey(i), data) for i, d in enumerate(devs)]
    flops = 2.0 * cfg["batch"] * cfg["m"] * cfg["n"] * cfg["k"]
    t0 = time.perf_counter()
    outs = [fn(a, b) for a, b in ops]
    jax.block_until_ready(outs)
    compile_s = time.perf_counter() - t0
    hlo = fn.lower(*ops[0]).compile().as_text()
    cc = [ln.strip() for ln in hlo.splitlines() if re.search(r"__cublas|__triton|__cudnn|triton_gemm", ln) and " = " in ln]
    for _ in range(3):
        outs = [fn(a, b) for a, b in ops]
    jax.block_until_ready(outs)

    time.sleep(idle)
    nb = 5
    t0 = time.perf_counter()
    for _ in range(nb):
        outs = [fn(a, b) for a, b in ops]
    jax.block_until_ready(outs)
    burst = (time.perf_counter() - t0) / nb

    calls, marks = 0, []
    t0 = time.perf_counter()
    while True:
        for _ in range(4):
            outs = [fn(a, b) for a, b in ops]
        jax.block_until_ready(outs)
        calls += 4
        now = time.perf_counter()
        marks.append((now, calls))
        if now - t0 >= sustain:
            break
    half = [m for m in marks[:-1] if m[0] - t0 >= sustain / 2]
    ta, ca = half[0] if half else (t0, 0)
    tb, cb = marks[-1]
    sustained = (tb - ta) / max(cb - ca, 1)
    tele = sampler.window(ta, tb, len(devs)) if sampler else None
    del outs, ops
    return dict(
        cfg,
        data=data,
        flops=flops,
        compile_s=round(compile_s, 2),
        burst_us=burst * 1e6,
        burst_pfs=flops / burst / 1e15,
        sustained_us=sustained * 1e6,
        sustained_pfs=flops / sustained / 1e15,
        sustained_calls=cb - ca,
        telemetry=tele,
        cublas_call=cc[0][:600] if cc else None,
    )


def profile_kernels(cfgs, dev, outdir):
    """Capture 3 calls of every config and return {config: [(kernel, us), ...]} from the xplane."""
    import glob
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "loop-260930-mfu30"))
    from overlap import load, plane_events  # noqa: E402

    fns = [(c, gemm_fn(c), operands(c, dev, jax.random.PRNGKey(7))) for c in cfgs]
    for _, fn, (a, b) in fns:
        jax.block_until_ready(fn(a, b))
    jax.profiler.start_trace(outdir)
    for _, fn, (a, b) in fns:
        for _ in range(3):
            out = fn(a, b)
        jax.block_until_ready(out)
    jax.profiler.stop_trace()
    path = sorted(glob.glob(f"{outdir}/**/*.xplane.pb", recursive=True))[-1]
    xs = load(path)
    planes = {p.name: p for p in xs.planes}
    res = {}
    for lname, name, s, e, st in plane_events(planes["/device:GPU:0"]):
        if not lname.startswith("Stream"):
            continue
        mod = st.get("hlo_module") or ""
        if mod.startswith("jit_gemm_"):
            res.setdefault(mod[len("jit_gemm_") :], []).append((name[:80], round((e - s) * 1e-6, 1)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--sustain", type=float, default=6.0)
    ap.add_argument("--idle", type=float, default=1.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--no-extra", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--data", default="normal", help="comma list of operand patterns: normal,zeros,mant2,sparse50,smooth")
    args = ap.parse_args()

    cfgs = json.load(open(args.configs))
    seen, uniq = {}, []
    for c in cfgs:
        if c["name"] in seen:
            seen[c["name"]]["insitu_t_step"] += c["insitu_t_step"]
            continue
        seen[c["name"]] = c
        uniq.append(c)
    if not args.no_extra:
        uniq += EXTRA
    if args.only:
        keep = set(args.only.split(","))
        uniq = [c for c in uniq if c["name"] in keep or any(c["name"].startswith(k) for k in keep)]

    devs = jax.local_devices()[: args.devices]
    print(f"devices={devs} XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}", flush=True)
    nvml = Nvml(len(jax.local_devices()))
    sampler = Sampler(nvml) if nvml.ok else None
    if sampler:
        sampler.start()
        time.sleep(0.5)
        print("idle telemetry", sampler.window(0, time.perf_counter(), args.devices), flush=True)
    results = []
    for c, data in [(c, d) for c in uniq for d in args.data.split(",")]:
        try:
            r = run_config(c, devs, sampler, args.sustain, args.idle, data)
        except Exception as e:  # keep going across configs; record the failure
            r = dict(c, data=data, error=f"{type(e).__name__}: {str(e)[:300]}")
        results.append(r)
        t = r.get("telemetry") or {}
        print(
            f"{c['name']:36s} {data:8s} burst {r.get('burst_pfs', 0):.3f} PF/s  sustained {r.get('sustained_pfs', 0):.3f} PF/s "
            f"({r.get('sustained_us', 0):8.0f} us)  clk {t.get('sm_clock_mhz', 0):6.0f} MHz  "
            f"pwr {t.get('power_w', 0):6.0f} W  reasons {t.get('clock_event_reasons')}  insitu {c.get('insitu_pfs')}"
            + (f"  ERROR {r['error']}" if "error" in r else ""),
            flush=True,
        )
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    if args.profile:
        kern = profile_kernels([c for c in uniq], devs[0], "/tmp/gemm_bench_profile")
        for r in results:
            r["kernels"] = kern.get(r["name"])
            print(f"{r['name']:36s} {kern.get(r['name'])}", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    if sampler:
        sampler.stop = True


if __name__ == "__main__":
    main()
