# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Time the hero expert MLP's grouped GEMM legs on QuACK and on PyTorch's grouped_mm, at hero shapes.

The ragged EP hero runs its expert MLP per chunk: one receiver buffer of ``capacity`` rows holding
``len(group_sizes)`` experts' accepted rows, expert-major, followed by unspecified padding rows.
Marin runs it on QuACK's SM100 grouped GEMMs (``_CuteExpertMlp``); OLMo-core 3 runs the same
BF16 math on ``torch.nn.functional.grouped_mm``, whose CUDA fast path is ATen's CUTLASS
``bf16bf16_grouped_mm`` (OLMo's explicit-output wrapper calls the same kernel), with SwiGLU as a
separate compiled elementwise kernel. This script times each leg of both, and both complete
operators, on identical operands, and checks every output against a float32 reference.

Legs, for ``M`` active rows, latent ``H`` and intermediate ``I``:

    fwd gate/up (+ SwiGLU)   [M, H] x [E, H, 2I]
    fwd down                 [M, I] x [E, I, H]
    bwd dh (+ dSwiGLU)       [M, H] x [E, H, I]^T     QuACK fuses the SwiGLU backward and <h, dh>
    bwd dx                   [M, 2I] x [E, 2I, H]
    wgrad w2                 [M, I]^T x [M, H] per group
    wgrad w13                [M, H]^T x [M, 2I] per group

Every operand is drawn once from ``numpy.random.default_rng(seed)``, rounded to bf16 in JAX, and
handed to PyTorch through DLPack, so both frameworks see the same bits. Rows past the active count
hold NaN, so a kernel that reads them poisons its own output. Each leg's inputs that come from an
earlier leg are the float32 reference's values rounded to bf16, identical for both
implementations, so each leg's error is its own.

Timing: ``--warmup`` untimed calls, then ``--reps`` calls each synchronized on the host (complete
operator time, including dispatch, allocation and any layout work inside the timed function),
then the same number of calls under the framework's profiler, each followed by a short host
sleep so that every call's kernels form one group on the GPU timeline (kernel time: the summed
duration of the device activities in that group).

Examples::

    python lib/levanter/scripts/bench/bench_expert_mlp_legs.py --probe
    python lib/levanter/scripts/bench/bench_expert_mlp_legs.py \
        --group-sizes 82298,94440,85356 --capacity 301466 --json /tmp/legs.json
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import glob
import gzip
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from levanter.grug._moe.common import _interleave_gate_up
from levanter.grug._moe.ep_ragged_all_to_all import _CuteExpertMlp, _RaggedDotExpertMlp
from levanter.grug._moe.quack_moe_cute import (
    quack_gated_grouped_gemm,
    quack_grouped_dswiglu_gemm,
    quack_grouped_gemm,
    quack_grouped_wgrad,
)
from levanter.grug._moe.sonic_cute import _QUACK_GATED_KW, _QUACK_GROUPED_KW, _QUACK_WGRAD_KW
from marin.profiling.xplane import _xspace_message_class

# One hero chunk at step 180050 (run m30pr-c6-01, layer 13, shard 22, chunk 1): the median
# active-row chunk of that step. See the issue record for how the group sizes were derived.
_DEFAULT_GROUP_SIZES = (82298, 94440, 85356)
_DEFAULT_CAPACITY = 301_466
_DEFAULT_HIDDEN = 3072
_DEFAULT_INTERMEDIATE = 3072
_PROFILE_SLEEP = 0.004
_PIPELINED_GAP = 0.05
_PIPELINED_BATCHES = 5
_GROUP_GAP_NS = 1_000_000
_OLMO_REV = "a800c06edcec8e4652de2f80fe0ea572d689fcc3"
_OLMO_SWIGLU_URL = f"https://raw.githubusercontent.com/allenai/olmo-core/{_OLMO_REV}/src/olmo_core/kernels/swiglu.py"


@dataclasses.dataclass
class Timing:
    variant: str
    leg: str
    flops: float
    op_ms: list[float]
    pipelined_ms: list[float]
    kernel_ms: list[float]
    pipelined_kernel_ms: float | None
    kernels: list[str]
    peak_bytes: int | None
    errors: dict[str, tuple[float, float]]

    def summary(self) -> dict[str, Any]:
        def stats(xs: list[float]) -> dict[str, float]:
            if not xs:
                return {}
            q = np.percentile(xs, [10, 50, 90])
            return dict(median=float(q[1]), p10=float(q[0]), p90=float(q[2]), min=min(xs), max=max(xs), n=len(xs))

        op = stats(self.op_ms)
        pipe = stats(self.pipelined_ms)
        kern = stats(self.kernel_ms)
        return dict(
            variant=self.variant,
            leg=self.leg,
            tflops_pipelined=self.flops / (pipe["median"] * 1e-3) / 1e12 if pipe else None,
            tflops_kernel=self.flops / (kern["median"] * 1e-3) / 1e12 if kern else None,
            op_ms=op,
            pipelined_ms=pipe,
            kernel_ms=kern,
            pipelined_kernel_ms=self.pipelined_kernel_ms,
            kernels=self.kernels,
            peak_bytes=self.peak_bytes,
            errors=self.errors,
        )


# --------------------------------------------------------------------------------------------
# Profiler parsing: group device activities into one group per call.


def _group(intervals: list[tuple[int, int, str]]) -> list[list[tuple[int, int, str]]]:
    groups: list[list[tuple[int, int, str]]] = []
    last_end = None
    for s, e, n in sorted(intervals):
        if last_end is None or s - last_end > _GROUP_GAP_NS:
            groups.append([])
        groups[-1].append((s, e, n))
        last_end = e if last_end is None else max(last_end, e)
    return groups


def _busy_ms(group: list[tuple[int, int, str]]) -> float:
    merged: list[list[int]] = []
    for s, e, _ in sorted(group):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged) * 1e-6


def _jax_device_intervals(trace_dir: str) -> list[tuple[int, int, str]]:
    paths = glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True)
    if len(paths) != 1:
        raise RuntimeError(f"expected one xplane under {trace_dir}, found {paths}")
    xspace = _xspace_message_class()()
    with open(paths[0], "rb") as fh:
        xspace.ParseFromString(fh.read())
    plane = next(p for p in xspace.planes if p.name == "/device:GPU:0")
    meta = {int(k): v for k, v in plane.event_metadata.items()}
    out = []
    for line in plane.lines:
        name = line.display_name or line.name
        # Derived lines ("XLA Modules", "XLA Ops", "Steps") repeat the stream events.
        if not name.startswith("Stream"):
            continue
        base_ns = line.timestamp_ns
        for ev in line.events:
            m = meta[int(ev.metadata_id)]
            start = base_ns + ev.offset_ps // 1000
            out.append((start, start + max(1, ev.duration_ps // 1000), m.display_name or m.name))
    return out


def _torch_device_intervals(trace_path: str) -> list[tuple[int, int, str]]:
    opener = gzip.open if trace_path.endswith(".gz") else open
    with opener(trace_path, "rt") as fh:
        trace = json.load(fh)
    out = []
    for ev in trace["traceEvents"]:
        if ev.get("ph") == "X" and ev.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            start = int(float(ev["ts"]) * 1000)
            out.append((start, start + max(1, int(float(ev["dur"]) * 1000)), ev["name"]))
    return out


# --------------------------------------------------------------------------------------------
# Timing drivers.


def _time_jax(fn: Callable, args: tuple, warmup: int, reps: int) -> dict[str, Any]:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    op_ms = []
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        op_ms.append((time.perf_counter() - start) * 1e3)
    pipelined_ms = [_jax_back_to_back(fn, args, reps) for _ in range(_PIPELINED_BATCHES)]
    with tempfile.TemporaryDirectory() as trace_dir:
        jax.profiler.start_trace(trace_dir)
        for _ in range(reps):
            jax.block_until_ready(fn(*args))
            time.sleep(_PROFILE_SLEEP)
        time.sleep(_PIPELINED_GAP)
        _jax_back_to_back(fn, args, reps)
        jax.profiler.stop_trace()
        groups = _group(_jax_device_intervals(trace_dir))
    return _timings(op_ms, pipelined_ms, groups, reps)


def _jax_back_to_back(fn: Callable, args: tuple, reps: int) -> float:
    """Mean wall time per call with the next call always queued behind the running one.

    At most two calls are in flight. An unbounded queue would hold every queued call's output
    until it ran, and the allocator would then stall the host for memory instead of the GPU
    running the kernels back to back.
    """
    start = time.perf_counter()
    previous = fn(*args)
    for _ in range(reps - 1):
        current = fn(*args)
        jax.block_until_ready(previous)
        previous = current
    jax.block_until_ready(previous)
    return (time.perf_counter() - start) * 1e3 / reps


def _time_torch(fn: Callable, args: tuple, warmup: int, reps: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    op_ms = []
    for _ in range(reps):
        start = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        op_ms.append((time.perf_counter() - start) * 1e3)
    pipelined_ms = []
    for _ in range(_PIPELINED_BATCHES):
        start = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        torch.cuda.synchronize()
        pipelined_ms.append((time.perf_counter() - start) * 1e3 / reps)
    with tempfile.TemporaryDirectory() as trace_dir:
        path = os.path.join(trace_dir, "trace.json")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                fn(*args)
                torch.cuda.synchronize()
                time.sleep(_PROFILE_SLEEP)
            time.sleep(_PIPELINED_GAP)
            for _ in range(reps):
                fn(*args)
            torch.cuda.synchronize()
        prof.export_chrome_trace(path)
        groups = _group(_torch_device_intervals(path))
    return _timings(op_ms, pipelined_ms, groups, reps)


def _timings(op_ms: list[float], pipelined_ms: list[float], groups: list, reps: int) -> dict[str, Any]:
    """Split the profiled window into its per-call groups and the final back-to-back group.

    The back-to-back calls run with no host gap, so they merge into one group, whose busy time per
    call is the steady-state kernel time under the power cap.
    """
    pipelined_kernel_ms = _busy_ms(groups[-1]) / reps if groups else None
    return dict(
        op_ms=op_ms,
        pipelined_ms=pipelined_ms,
        kernel_ms=_kernel_ms(groups[:-1], reps),
        pipelined_kernel_ms=pipelined_kernel_ms,
        kernels=_kernel_names(groups[:-1]),
    )


def _kernel_ms(groups: list, reps: int) -> list[float]:
    if len(groups) != reps:
        # PyTorch's profiler can miss the first call after it starts; report the mean of the rest.
        print(f"    warning: {len(groups)} kernel groups for {reps} calls; kernel time is their mean")
        return [sum(_busy_ms(g) for g in groups) / max(1, len(groups))]
    return [_busy_ms(g) for g in groups]


def _kernel_names(groups: list) -> list[str]:
    if not groups:
        return []
    seen: dict[str, None] = {}
    for _, _, name in sorted(groups[len(groups) // 2]):
        seen.setdefault(name, None)
    return list(seen)


# --------------------------------------------------------------------------------------------
# Numerics.


def _errors(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float]:
    """max and mean |got - want|, relative to mean |want|; NaN anywhere makes both NaN."""
    delta = (got.float() - want.float()).abs()
    scale = want.float().abs().mean()
    return float(delta.max() / scale), float(delta.mean() / scale)


def _t(x: jax.Array) -> torch.Tensor:
    return torch.from_dlpack(jax.block_until_ready(x))


def _j(x: torch.Tensor) -> jax.Array:
    return jnp.from_dlpack(x.contiguous())


def _interleave_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack([a, b], dim=-1).reshape(*a.shape[:-1], 2 * a.shape[-1])


def _deinterleave_t(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[..., 0::2], x[..., 1::2]


def _with_nan_tail(x: torch.Tensor, active: int) -> torch.Tensor:
    x = x.clone()
    x[active:] = float("nan")
    return x


def _peak_jax(fn, args) -> int:
    analysis = fn.lower(*args).compile().memory_analysis()
    return int(analysis.temp_size_in_bytes + analysis.output_size_in_bytes)


def _peak_torch(fn, args) -> int:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn(*args)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out
    return int(peak)


# --------------------------------------------------------------------------------------------
# Versions and the PyTorch fast-path check.


def _versions() -> dict[str, Any]:
    pkgs = {}
    for name in (
        "jax",
        "jaxlib",
        "jax-cuda13-plugin",
        "jax-cuda13-pjrt",
        "quack-kernels",
        "nvidia-cutlass-dsl",
        "torch",
        "triton",
        "nvidia-cublas",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu13",
        "nvidia-cudnn-cu12",
        "nvidia-cuda-runtime",
        "nvidia-cuda-runtime-cu12",
        "nvidia-nccl-cu13",
    ):
        try:
            pkgs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,power.limit,clocks.max.sm,memory.total",
            "--format=csv,noheader",
            "-i",
            "0",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return dict(
        packages=pkgs,
        torch_cuda=torch.version.cuda,
        torch_cudnn=torch.backends.cudnn.version(),
        cublaslt_runtime=_cublaslt_versions(),
        jax_device=jax.devices()[0].device_kind,
        torch_device=torch.cuda.get_device_name(0),
        torch_capability=torch.cuda.get_device_capability(0),
        nvidia_smi=smi,
        python=platform.python_version(),
        machine=platform.machine(),
    )


def _cublaslt_versions() -> dict[str, int | str]:
    out: dict[str, int | str] = {}
    for soname in ("libcublasLt.so.12", "libcublasLt.so.13"):
        try:
            out[soname] = int(ctypes.CDLL(soname).cublasLtGetVersion())
        except OSError as exc:
            out[soname] = f"not loadable: {exc}"[:80]
    return out


def _check_torch_fast_path(device: torch.device) -> list[str]:
    """Run grouped_mm at unaligned group sizes with sync errors on and return its kernel names.

    The fallback copies ``offs`` to the host and loops over groups with ``mm``; under
    ``set_sync_debug_mode("error")`` that copy raises, so returning at all rules it out.
    """
    sizes = torch.tensor([1037, 2049, 1531], dtype=torch.int32, device=device)
    offs = torch.cumsum(sizes, 0, dtype=torch.int32)
    a = torch.randn(4800, 512, device=device, dtype=torch.bfloat16)
    b = torch.randn(3, 512, 1024, device=device, dtype=torch.bfloat16)
    F.grouped_mm(a, b, offs=offs)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        torch.cuda.set_sync_debug_mode("error")
        try:
            out = F.grouped_mm(a, b, offs=offs)
            wg = F.grouped_mm(a[:4617].t(), torch.randn(4617, 256, device=device, dtype=torch.bfloat16), offs=offs)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    want = torch.cat([a[s:e].float() @ b[i].float() for i, (s, e) in enumerate(zip([0, 1037, 3086], offs.tolist()))])
    err = _errors(out[: int(offs[-1])], want)
    names = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
    print(f"  torch grouped_mm fast path: no host sync, rel err {err[0]:.2e}; wgrad shape {tuple(wg.shape)}")
    for n in names:
        print(f"    kernel: {n[:150]}")
    return names


def _ensure_triton_cache_dir() -> None:
    """Point Triton at a writable cache; its driver and inductor both refuse to start without one."""
    import triton  # noqa: PLC0415  (only the PyTorch variant needs Triton)

    before = dict(triton.knobs.cache.__dict__)
    triton.knobs.cache.dir = os.environ.get("TRITON_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "triton-cache")
    print(f"  triton cache knobs before: {before}; dir now {triton.knobs.cache.dir}")


def _load_olmo_swiglu():
    path = os.path.join(tempfile.gettempdir(), f"olmo_swiglu_{_OLMO_REV[:7]}.py")
    if not os.path.exists(path):
        subprocess.run(["curl", "-sfL", "-o", path, _OLMO_SWIGLU_URL], check=True)
    spec = importlib.util.spec_from_file_location("olmo_swiglu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------------
# The two implementations.


def _olmo_swiglu_fwd(up_gate: torch.Tensor) -> torch.Tensor:
    # RoutedExperts.chunk_and_activate (olmo_core/nn/moe/v2/routed_experts.py@a800c06), autograd path.
    up, gate = up_gate.chunk(2, dim=-1)
    return up * F.silu(gate)


def _olmo_swiglu_bwd(up_gate: torch.Tensor, grad_h: torch.Tensor) -> torch.Tensor:
    # _swiglu_backward_grad_up_gate_impl (routed_experts.py@a800c06), which OLMo compiles.
    hidden = up_gate.shape[-1] // 2
    up = up_gate[:, :hidden].float()
    gate = up_gate[:, hidden:].float()
    grad_h = grad_h.float()
    sig = torch.sigmoid(gate)
    silu_gate = gate * sig
    dsilu = sig * (1.0 + gate * (1.0 - sig))
    return torch.cat((grad_h * silu_gate, grad_h * up * dsilu), dim=-1).to(up_gate.dtype)


def _olmo_swiglu_bwd_row_dot(up_gate: torch.Tensor, grad_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """OLMo's SwiGLU backward plus Marin's per-row ``<h, dh>`` (the routing-weight gradient)."""
    hidden = up_gate.shape[-1] // 2
    up = up_gate[:, :hidden].float()
    gate = up_gate[:, hidden:].float()
    grad_h = grad_h.float()
    sig = torch.sigmoid(gate)
    silu_gate = gate * sig
    dsilu = sig * (1.0 + gate * (1.0 - sig))
    d_up_gate = torch.cat((grad_h * silu_gate, grad_h * up * dsilu), dim=-1).to(up_gate.dtype)
    return d_up_gate, (silu_gate * up * grad_h).sum(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group-sizes", type=str, default=",".join(map(str, _DEFAULT_GROUP_SIZES)))
    parser.add_argument("--capacity", type=int, default=_DEFAULT_CAPACITY)
    parser.add_argument("--hidden", type=int, default=_DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=_DEFAULT_INTERMEDIATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--variants", type=str, default="quack,xla", help="quack,xla or torch")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--probe", action="store_true", help="print versions and check the torch fast path")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", 0)
    variants = set(args.variants.split(","))
    # PyTorch's profiler and JAX's both subscribe to CUPTI, so one process profiles one framework:
    # run the JAX variants and the PyTorch variant as separate invocations with the same seed.
    if not args.probe and "torch" in variants and variants & {"quack", "xla"}:
        raise ValueError("profile PyTorch in its own invocation: --variants torch")
    torch.zeros(1, device=device)
    versions = _versions()
    print(json.dumps(versions, indent=1))
    fast_path_kernels = _check_torch_fast_path(device) if "torch" in variants or args.probe else []
    if args.probe:
        print({k: os.environ.get(k) for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "USER", "HOME")})
        _ensure_triton_cache_dir()
        compiled = torch.compile(_olmo_swiglu_fwd, fullgraph=True, dynamic=False)
        print("compiled swiglu ok", compiled(torch.randn(64, 256, device=device, dtype=torch.bfloat16)).shape)
        return 0

    sizes = [int(s) for s in args.group_sizes.split(",")]
    E, C, H, I = len(sizes), args.capacity, args.hidden, args.intermediate
    M = sum(sizes)
    if M > C:
        raise ValueError(f"group sizes sum to {M} > capacity {C}")
    print(f"shape: E={E} C={C} H={H} I={I} active M={M} group_sizes={sizes} seed={args.seed}")

    # ---- operands, drawn once and shared by every implementation.
    rng = np.random.default_rng(args.seed)
    x_np = rng.standard_normal((C, H), dtype=np.float32)
    w_gate_np = rng.standard_normal((E, H, I), dtype=np.float32) / np.sqrt(H)
    w_up_np = rng.standard_normal((E, H, I), dtype=np.float32) / np.sqrt(H)
    w2_np = rng.standard_normal((E, I, H), dtype=np.float32) / np.sqrt(I)
    dy_np = rng.standard_normal((C, H), dtype=np.float32)
    x_np[M:] = np.nan
    dy_np[M:] = np.nan
    x = jnp.asarray(x_np, jnp.bfloat16)
    dy = jnp.asarray(dy_np, jnp.bfloat16)
    w_gate = jnp.asarray(w_gate_np, jnp.bfloat16)
    w_up = jnp.asarray(w_up_np, jnp.bfloat16)
    w2 = jnp.asarray(w2_np, jnp.bfloat16)
    del x_np, dy_np, w_gate_np, w_up_np, w2_np
    moe_w13 = jnp.concatenate([w_gate, w_up], axis=-1)  # grug layout [E, H, 2I], gate then up
    w13_il = jax.block_until_ready(_interleave_gate_up(moe_w13, I))  # QuACK layout
    group_sizes = jnp.asarray(sizes, jnp.int32)
    cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes).astype(jnp.int32)])
    physical = group_sizes.at[-1].add(C - M)

    xt, dyt = _t(x), _t(dy)
    wgt, wut, w2t = _t(w_gate), _t(w_up), _t(w2)
    # OLMo layout: w_up_gate [E, 2I, H] with up rows first; grouped_mm sees its transpose.
    w_up_gate_t = torch.cat([wut.transpose(1, 2), wgt.transpose(1, 2)], dim=1).contiguous()
    offs = torch.cumsum(torch.tensor(sizes, dtype=torch.int32, device=device), 0, dtype=torch.int32)
    bounds = [0, *np.cumsum(sizes).tolist()]

    # ---- float32 reference, per group, TF32 off.
    print("building float32 reference")

    def per_group(fn):
        return torch.cat([fn(bounds[i], bounds[i + 1], i) for i in range(E)])

    ref = {}
    ref["g"] = per_group(lambda s, e, i: xt[s:e].float() @ wgt[i].float())
    ref["u"] = per_group(lambda s, e, i: xt[s:e].float() @ wut[i].float())
    ref["h"] = F.silu(ref["g"]) * ref["u"]
    ref["y"] = per_group(lambda s, e, i: ref["h"][s:e] @ w2t[i].float())
    # Leg inputs that come from an earlier leg: the reference rounded to bf16, identical for all.
    g_c, u_c, h_c = ref["g"].bfloat16(), ref["u"].bfloat16(), ref["h"].bfloat16()
    ref["y_leg"] = per_group(lambda s, e, i: h_c[s:e].float() @ w2t[i].float())
    ref["dh"] = per_group(lambda s, e, i: dyt[s:e].float() @ w2t[i].float().T)
    sg = torch.sigmoid(g_c.float())
    silu_c = g_c.float() * sg
    ref["dgate_leg"] = ref["dh"] * u_c.float() * (sg + silu_c * (1 - sg))
    ref["dup_leg"] = ref["dh"] * silu_c
    ref["rowdot_leg"] = (silu_c * u_c.float() * ref["dh"]).sum(-1)
    dgate_c, dup_c = ref["dgate_leg"].bfloat16(), ref["dup_leg"].bfloat16()
    ref["dx_leg"] = per_group(
        lambda s, e, i: dgate_c[s:e].float() @ wgt[i].float().T + dup_c[s:e].float() @ wut[i].float().T
    )
    ref["dw2_leg"] = torch.stack([h_c[s:e].float().T @ dyt[s:e].float() for s, e in zip(bounds, bounds[1:])])
    ref["dwgate_leg"] = torch.stack([xt[s:e].float().T @ dgate_c[s:e].float() for s, e in zip(bounds, bounds[1:])])
    ref["dwup_leg"] = torch.stack([xt[s:e].float().T @ dup_c[s:e].float() for s, e in zip(bounds, bounds[1:])])
    # Unrounded chain for the complete operator.
    sg = torch.sigmoid(ref["g"])
    silu = ref["g"] * sg
    dgate = ref["dh"] * ref["u"] * (sg + silu * (1 - sg))
    dup = ref["dh"] * silu
    ref["rowdot"] = (ref["h"] * ref["dh"]).sum(-1)
    ref["dx"] = per_group(lambda s, e, i: dgate[s:e] @ wgt[i].float().T + dup[s:e] @ wut[i].float().T)
    ref["dw2"] = torch.stack([ref["h"][s:e].T @ dyt[s:e].float() for s, e in zip(bounds, bounds[1:])])
    ref["dwgate"] = torch.stack([xt[s:e].float().T @ dgate[s:e] for s, e in zip(bounds, bounds[1:])])
    ref["dwup"] = torch.stack([xt[s:e].float().T @ dup[s:e] for s, e in zip(bounds, bounds[1:])])
    del sg, silu, dgate, dup, silu_c
    torch.cuda.synchronize()

    pad = C - M

    def padded(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([t, torch.full((pad, *t.shape[1:]), float("nan"), dtype=t.dtype, device=t.device)])

    # Common leg inputs in each layout.
    gu_il_c = _j(padded(_interleave_t(g_c, u_c)))
    h_cj = _j(padded(h_c))
    dgu_il_c = _j(padded(_interleave_t(dgate_c, dup_c)))
    up_gate_c = padded(torch.cat([u_c, g_c], dim=-1))
    h_ct = padded(h_c)
    d_up_gate_c = padded(torch.cat([dup_c, dgate_c], dim=-1))
    del g_c, u_c, h_c, dgate_c, dup_c

    flops = {
        "fwd_gate_up": 4.0 * M * H * I,
        "fwd_down": 2.0 * M * I * H,
        "bwd_dh": 2.0 * M * H * I,
        "bwd_dx": 4.0 * M * I * H,
        "wgrad_w2": 2.0 * M * I * H,
        "wgrad_w13": 4.0 * M * H * I,
        "swiglu_fwd": 0.0,
        "swiglu_bwd": 0.0,
    }
    flops["mlp_fwd_bwd"] = sum(v for k, v in flops.items() if not k.startswith("swiglu"))
    results: list[Timing] = []

    def run(variant, leg, timer, fn, fn_args, errors_fn, peak_fn):
        print(f"  {variant:28s} {leg:14s}", end="", flush=True)
        out = fn(*fn_args)
        if timer is _time_jax:
            out = jax.block_until_ready(out)
        else:
            torch.cuda.synchronize()
        errs = errors_fn(out)
        del out
        peak = peak_fn(fn, fn_args)
        tm = timer(fn, fn_args, args.warmup, args.reps)
        t = Timing(
            variant,
            leg,
            flops.get(leg, 0.0),
            tm["op_ms"],
            tm["pipelined_ms"],
            tm["kernel_ms"],
            tm["pipelined_kernel_ms"],
            tm["kernels"],
            peak,
            errs,
        )
        s = t.summary()
        print(
            f" op {s['op_ms']['median']:7.3f} [{s['op_ms']['p10']:.3f},{s['op_ms']['p90']:.3f}]"
            f" pipelined {s['pipelined_ms']['median']:7.3f} [{s['pipelined_ms']['min']:.3f},{s['pipelined_ms']['max']:.3f}]"
            f" kernel {s['kernel_ms'].get('median', float('nan')):7.3f} steady {s['pipelined_kernel_ms'] or 0:7.3f} ms"
            f" {s['tflops_pipelined'] or 0:6.0f} TF/s peak {peak / 2**30:5.2f} GiB "
            + " ".join(f"{k}={v[0]:.2e}" for k, v in errs.items())
        )
        results.append(t)

    def act(t: torch.Tensor) -> torch.Tensor:
        return t[:M]

    # ---------------------------------------------------------------- QuACK (shipped config)
    if "quack" in variants:
        print("QuACK (sonic_cute shipped tile/cluster/CLC settings)")
        v = "quack"
        fwd_gu = jax.jit(lambda x, w, cu: quack_gated_grouped_gemm(x, w, cu, return_preact=True, **_QUACK_GATED_KW))

        def err_gu(out):
            gu, h = out
            g, u = _deinterleave_t(act(_t(gu)))
            return dict(gate=_errors(g, ref["g"]), up=_errors(u, ref["u"]), h=_errors(act(_t(h)), ref["h"]))

        run(v, "fwd_gate_up", _time_jax, fwd_gu, (x, w13_il, cu), err_gu, _peak_jax)
        fwd_down = jax.jit(lambda h, w, cu: quack_grouped_gemm(h, w, cu, b_major="n", **_QUACK_GROUPED_KW))
        run(
            v,
            "fwd_down",
            _time_jax,
            fwd_down,
            (h_cj, w2, cu),
            lambda o: dict(y=_errors(act(_t(o)), ref["y_leg"])),
            _peak_jax,
        )
        bwd_dh = jax.jit(lambda dy, w, gu, cu: quack_grouped_dswiglu_gemm(dy, w, gu, cu, **_QUACK_GROUPED_KW))

        def err_dh(out):
            dgu, rowdot = out
            dg, du = _deinterleave_t(act(_t(dgu)))
            return dict(
                dgate=_errors(dg, ref["dgate_leg"]),
                dup=_errors(du, ref["dup_leg"]),
                rowdot=_errors(act(_t(rowdot)), ref["rowdot_leg"]),
            )

        run(v, "bwd_dh", _time_jax, bwd_dh, (dy, w2, gu_il_c, cu), err_dh, _peak_jax)
        bwd_dx = jax.jit(lambda d, w, cu: quack_grouped_gemm(d, w, cu, b_major="k", **_QUACK_GROUPED_KW))
        run(
            v,
            "bwd_dx",
            _time_jax,
            bwd_dx,
            (dgu_il_c, w13_il, cu),
            lambda o: dict(dx=_errors(act(_t(o)), ref["dx_leg"])),
            _peak_jax,
        )
        wgrad = jax.jit(lambda a, b, cu: quack_grouped_wgrad(a, b, cu, **_QUACK_WGRAD_KW))
        run(
            v,
            "wgrad_w2",
            _time_jax,
            wgrad,
            (h_cj, dy, cu),
            lambda o: dict(dw2=_errors(_t(o), ref["dw2_leg"])),
            _peak_jax,
        )

        def err_w13(o):
            dg, du = _deinterleave_t(_t(o))
            return dict(dwgate=_errors(dg, ref["dwgate_leg"]), dwup=_errors(du, ref["dwup_leg"]))

        run(v, "wgrad_w13", _time_jax, wgrad, (x, dgu_il_c, cu), err_w13, _peak_jax)

        mlp = _CuteExpertMlp()

        def quack_full(x, w13, w2, physical, active, dy):
            out, res = mlp.forward(x, w13, w2, physical, active)
            dx, dw13, dw2, rowdot = mlp.backward(res, dy)
            return out, dx, dw13, dw2, rowdot

        def err_full(o):
            out, dx, dw13, dw2, rowdot = o
            dw13 = _t(dw13)
            return dict(
                y=_errors(act(_t(out)), ref["y"]),
                dx=_errors(act(_t(dx)), ref["dx"]),
                dwgate=_errors(dw13[..., :I], ref["dwgate"]),
                dwup=_errors(dw13[..., I:], ref["dwup"]),
                dw2=_errors(_t(dw2), ref["dw2"]),
                rowdot=_errors(act(_t(rowdot)), ref["rowdot"]),
            )

        full_args = (x, moe_w13, w2, physical, group_sizes, dy)
        run(v, "mlp_fwd_bwd", _time_jax, jax.jit(quack_full), full_args, err_full, _peak_jax)

    # ---------------------------------------------------------------- PyTorch grouped_mm (OLMo-core 3 BF16 path)
    if "torch" in variants:
        print(f"PyTorch {torch.__version__} grouped_mm (OLMo-core {_OLMO_REV[:7]} BF16 routed-experts path)")
        v = "torch_grouped_mm"
        _ensure_triton_cache_dir()
        swiglu_fwd = torch.compile(_olmo_swiglu_fwd, fullgraph=True, dynamic=False)
        swiglu_bwd = torch.compile(_olmo_swiglu_bwd, fullgraph=True, dynamic=False)
        swiglu_bwd_dot = torch.compile(_olmo_swiglu_bwd_row_dot, fullgraph=True, dynamic=False)
        try:
            swiglu_fwd(up_gate_c[:64])
            swiglu_bwd(up_gate_c[:64], h_ct[:64])
            swiglu_bwd_dot(up_gate_c[:64], h_ct[:64])
        except Exception as exc:  # inductor needs a working Triton toolchain; record and run eager instead
            print(f"  torch.compile failed ({type(exc).__name__}: {str(exc)[:200]}); SwiGLU runs eager")
            traceback.print_exc()
            v = "torch_grouped_mm_eager_swiglu"
            swiglu_fwd, swiglu_bwd, swiglu_bwd_dot = _olmo_swiglu_fwd, _olmo_swiglu_bwd, _olmo_swiglu_bwd_row_dot
        w_down = w2t  # OLMo w_down [E, I, H] row-major, same as grug

        def t_gate_up(x):
            return F.grouped_mm(x, w_up_gate_t.transpose(1, 2), offs=offs)

        def err_t_gu(o):
            return dict(up=_errors(act(o)[:, :I], ref["u"]), gate=_errors(act(o)[:, I:], ref["g"]))

        run(v, "fwd_gate_up", _time_torch, t_gate_up, (xt,), err_t_gu, _peak_torch)
        run(
            v,
            "swiglu_fwd",
            _time_torch,
            swiglu_fwd,
            (up_gate_c,),
            lambda o: dict(h=_errors(act(o), ref["h"])),
            _peak_torch,
        )
        try:
            olmo_swiglu = _load_olmo_swiglu()
            n_valid = torch.tensor(M, device=device)

            def valid_prefix(ug):
                with torch.no_grad():
                    return olmo_swiglu.swiglu_valid_prefix(ug, n_valid)

            run(
                "olmo_triton",
                "swiglu_fwd",
                _time_torch,
                valid_prefix,
                (up_gate_c,),
                lambda o: dict(h=_errors(act(o), ref["h"])),
                _peak_torch,
            )
        except Exception as exc:  # the OLMo kernel is an optional extra row; report why it is missing
            print(f"  olmo swiglu_valid_prefix unavailable: {type(exc).__name__}: {exc}")
        run(
            v,
            "fwd_down",
            _time_torch,
            lambda h: F.grouped_mm(h, w_down, offs=offs),
            (h_ct,),
            lambda o: dict(y=_errors(act(o), ref["y_leg"])),
            _peak_torch,
        )
        run(
            v,
            "bwd_dh",
            _time_torch,
            lambda g: F.grouped_mm(g, w_down.transpose(1, 2), offs=offs),
            (dyt,),
            lambda o: dict(dh=_errors(act(o), ref["dh"])),
            _peak_torch,
        )
        dh_c = padded(ref["dh"].bfloat16())

        def err_sbwd(o):
            d_ug, rowdot = o
            return dict(
                dup=_errors(act(d_ug)[:, :I], ref["dup_leg"]),
                dgate=_errors(act(d_ug)[:, I:], ref["dgate_leg"]),
                rowdot=_errors(act(rowdot), ref["rowdot_leg"]),
            )

        run(v, "swiglu_bwd", _time_torch, swiglu_bwd_dot, (up_gate_c, dh_c), err_sbwd, _peak_torch)
        run(
            v + "_norowdot",
            "swiglu_bwd",
            _time_torch,
            swiglu_bwd,
            (up_gate_c, dh_c),
            lambda o: dict(dup=_errors(act(o)[:, :I], ref["dup_leg"]), dgate=_errors(act(o)[:, I:], ref["dgate_leg"])),
            _peak_torch,
        )
        del dh_c
        run(
            v,
            "bwd_dx",
            _time_torch,
            lambda d: F.grouped_mm(d, w_up_gate_t, offs=offs),
            (d_up_gate_c,),
            lambda o: dict(dx=_errors(act(o), ref["dx_leg"])),
            _peak_torch,
        )
        run(
            v,
            "wgrad_w2",
            _time_torch,
            lambda h, g: F.grouped_mm(h.transpose(0, 1), g, offs=offs),
            (h_ct, dyt),
            lambda o: dict(dw2=_errors(o, ref["dw2_leg"])),
            _peak_torch,
        )

        def err_t_w13(o):  # OLMo layout [E, 2I, H]
            return dict(
                dwup=_errors(o[:, :I].transpose(1, 2), ref["dwup_leg"]),
                dwgate=_errors(o[:, I:].transpose(1, 2), ref["dwgate_leg"]),
            )

        run(
            v,
            "wgrad_w13",
            _time_torch,
            lambda g, x: F.grouped_mm(g.transpose(0, 1), x, offs=offs),
            (d_up_gate_c, xt),
            err_t_w13,
            _peak_torch,
        )

        def err_t_w13_grug(o):  # grug layout [E, H, 2I]
            return dict(dwup=_errors(o[..., :I], ref["dwup_leg"]), dwgate=_errors(o[..., I:], ref["dwgate_leg"]))

        run(
            v + "_grug_layout",
            "wgrad_w13",
            _time_torch,
            lambda x, g: F.grouped_mm(x.transpose(0, 1), g, offs=offs),
            (xt, d_up_gate_c),
            err_t_w13_grug,
            _peak_torch,
        )

        def torch_full(x, w_up_gate, w_down, dy):
            up_gate = F.grouped_mm(x, w_up_gate.transpose(1, 2), offs=offs)
            h = swiglu_fwd(up_gate)
            y = F.grouped_mm(h, w_down, offs=offs)
            dh = F.grouped_mm(dy, w_down.transpose(1, 2), offs=offs)
            d_up_gate, rowdot = swiglu_bwd_dot(up_gate, dh)
            dx = F.grouped_mm(d_up_gate, w_up_gate, offs=offs)
            dw_down = F.grouped_mm(h.transpose(0, 1), dy, offs=offs)
            dw_up_gate = F.grouped_mm(d_up_gate.transpose(0, 1), x, offs=offs)
            return y, dx, dw_up_gate, dw_down, rowdot

        def err_t_full(o):
            y, dx, dwug, dw2, rowdot = o
            return dict(
                y=_errors(act(y), ref["y"]),
                dx=_errors(act(dx), ref["dx"]),
                dwgate=_errors(dwug[:, I:].transpose(1, 2), ref["dwgate"]),
                dwup=_errors(dwug[:, :I].transpose(1, 2), ref["dwup"]),
                dw2=_errors(dw2, ref["dw2"]),
                rowdot=_errors(act(rowdot), ref["rowdot"]),
            )

        run(v, "mlp_fwd_bwd", _time_torch, torch_full, (xt, w_up_gate_t, w_down, dyt), err_t_full, _peak_torch)

        # The JAX<->PyTorch boundary an eager integration would pay per call: DLPack both ways.
        def roundtrip(a):
            return _j(torch.from_dlpack(a))

        for _ in range(args.warmup):
            jax.block_until_ready(roundtrip(x))
        op_ms = []
        for _ in range(args.reps):
            start = time.perf_counter()
            jax.block_until_ready(roundtrip(x))
            op_ms.append((time.perf_counter() - start) * 1e3)
        print(f"  dlpack jax->torch->jax [{C},{H}] bf16: median {statistics.median(op_ms) * 1e3:.1f} us")
        results.append(Timing("dlpack", "roundtrip", 0.0, op_ms, [], [], None, [], None, {}))

    # ---------------------------------------------------------------- XLA ragged_dot (stock PJRT)
    if "xla" in variants:
        print("XLA ragged_dot (_RaggedDotExpertMlp, stock lowering of the pinned PJRT plugin)")
        mlp_xla = _RaggedDotExpertMlp(jax.nn.silu)

        def xla_full(x, w13, w2, physical, active, dy):
            out, res = mlp_xla.forward(x, w13, w2, physical, active)
            dx, dw13, dw2, rowdot = mlp_xla.backward(res, dy)
            return out, dx, dw13, dw2, rowdot

        def err_x_full(o):
            out, dx, dw13, dw2, rowdot = o
            dw13 = _t(dw13)
            return dict(
                y=_errors(act(_t(out)), ref["y"]),
                dx=_errors(act(_t(dx)), ref["dx"]),
                dwgate=_errors(dw13[..., :I], ref["dwgate"]),
                dwup=_errors(dw13[..., I:], ref["dwup"]),
                dw2=_errors(_t(dw2), ref["dw2"]),
                rowdot=_errors(act(_t(rowdot)), ref["rowdot"]),
            )

        run(
            "xla_ragged_dot",
            "mlp_fwd_bwd",
            _time_jax,
            jax.jit(xla_full),
            (x, moe_w13, w2, physical, group_sizes, dy),
            err_x_full,
            _peak_jax,
        )

    if "quack" in variants:
        print("QuACK complete operator again (drift check)")
        run("quack_repeat", "mlp_fwd_bwd", _time_jax, jax.jit(quack_full), full_args, err_full, _peak_jax)

    record = dict(
        config=dict(
            group_sizes=sizes,
            capacity=C,
            hidden=H,
            intermediate=I,
            active_rows=M,
            seed=args.seed,
            warmup=args.warmup,
            reps=args.reps,
            quack_gated_kw=repr(_QUACK_GATED_KW),
            quack_grouped_kw=repr(_QUACK_GROUPED_KW),
            quack_wgrad_kw=repr(_QUACK_WGRAD_KW),
            olmo_rev=_OLMO_REV,
        ),
        versions=versions,
        torch_fast_path_kernels=fast_path_kernels,
        results=[r.summary() for r in results],
    )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=1)
    print("JSON " + json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
