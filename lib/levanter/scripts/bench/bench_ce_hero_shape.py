# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU fused cross-entropy benchmark at the Grug MoE EP64 hero shape.

Per-rank CE shape for ``experiments/grug/moe_hero_ep`` HERO_MODEL is
``B=65_536`` tokens, ``H=6_144``, ``V=128_256`` with bf16 activations/weights and
an fp32 logit accumulator.  That is ~4 GB of tensors, so it fits on one GPU and
can be measured without the 0.78-1.3%% rack placement noise floor.

For every variant this reports:
  * forward wall time, backward wall time (total minus forward)
  * achieved TFLOP/s under both accounting conventions (see ``--help``)
  * peak HBM bytes
  * ``loss_diff`` and ``grad_max_abs_diff`` against a chunked float32 reference
  * whether the forward is bitwise identical to the current production path

Run one variant per process so ``peak_bytes_in_use`` is not polluted by a
previous variant; ``--fanout`` does that automatically across the local GPUs.

Examples::

    # everything, fanned out over the local GPUs
    python lib/levanter/scripts/bench/bench_ce_hero_shape.py --fanout

    # a single variant in this process
    python lib/levanter/scripts/bench/bench_ce_hero_shape.py --variants fast
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from levanter.kernels.pallas.fused_cross_entropy_loss.batched_xla import (
    linear_softmax_cross_entropy_loss_batched_xla,
)
from levanter.kernels.pallas.fused_cross_entropy_loss.config import BlockSizes
from levanter.kernels.pallas.fused_cross_entropy_loss.xla import (
    linear_softmax_cross_entropy_loss_xla,
)

RESULT_PREFIX = "RESULT_JSON "

# experiments/grug/moe_hero_ep/heuristic.py HERO_MODEL, mesh (1, 1, 64, 1):
# `data` and `model` are size 1, so the CE shard is the whole per-rank batch.
HERO_B = 65_536
HERO_H = 6_144
HERO_V = 128_256
# experiments/grug/moe_hero_ep/model.py:61 `_CE_BLOCK_SIZES`
HERO_V_BLOCK = 4_096
# BlockSizes.b_block_size default, left untouched by _CE_BLOCK_SIZES
HERO_B_BLOCK = 1_024


@dataclass(frozen=True)
class Variant:
    name: str
    impl: str = "xla"  # "xla" | "batched_xla"
    b_block_size: int = HERO_B_BLOCK
    v_block_size: int = HERO_V_BLOCK
    fast_backward: bool = False
    bwd_batch_block_size: Optional[int] = None
    bwd_v_block_size: Optional[int] = None
    note: str = ""


VARIANTS: tuple[Variant, ...] = (
    Variant("baseline", note="today: implementation='xla', _CE_BLOCK_SIZES, nested fori_loop backward"),
    # --- the ladder, one rung at a time ---
    Variant("fast", fast_backward=True, note="scan + dense one-hot + bf16 dlogits, untiled batch"),
    Variant("fast-bwdv2048", fast_backward=True, bwd_v_block_size=2048),
    Variant("fast-bwdv8192", fast_backward=True, bwd_v_block_size=8192),
    Variant("fast-bwdv16384", fast_backward=True, bwd_v_block_size=16384),
    Variant("fast-bwdv32768", fast_backward=True, bwd_v_block_size=32768),
    # --- peak-memory / speed trade: keep a batch tile in the backward ---
    Variant("fast-bb8192", fast_backward=True, bwd_batch_block_size=8192),
    Variant("fast-bb16384", fast_backward=True, bwd_batch_block_size=16384),
    # --- does the forward's own batch blocking still cost anything? ---
    Variant("baseline-fwdfull", b_block_size=HERO_B, note="forward batch blocking off, old backward"),
    Variant("fast-fwdfull", b_block_size=HERO_B, fast_backward=True),
    # --- maximum launch-count reduction: single-shot forward + 8-iteration backward ---
    Variant(
        "fast-max",
        b_block_size=HERO_B,
        fast_backward=True,
        bwd_v_block_size=16384,
        note="fwd 32 iters + bwd 8 iters (vs 2048 + 2048 at baseline)",
    ),
    Variant(
        "fast-max32k",
        b_block_size=HERO_B,
        fast_backward=True,
        bwd_v_block_size=32768,
        note="fwd 32 iters + bwd 4 iters",
    ),
    # --- the GB10 ladder implementation as shipped, unmodified ---
    Variant("batched_xla", impl="batched_xla", note="existing GB10-keyed ladder; may raise on GB200"),
)
VARIANTS_BY_NAME = {v.name: v for v in VARIANTS}


@dataclass
class Result:
    variant: str
    ok: bool = True
    error: str = ""
    device_kind: str = ""
    b: int = HERO_B
    h: int = HERO_H
    v: int = HERO_V
    b_block_size: int = 0
    v_block_size: int = 0
    bwd_v_block_size: Optional[int] = None
    bwd_batch_block_size: Optional[int] = None
    compile_fwd_s: float = 0.0
    compile_total_s: float = 0.0
    fwd_s: float = 0.0
    total_s: float = 0.0
    bwd_s: float = 0.0
    # 1 GEMM forward.
    fwd_tflops: float = 0.0
    # 3 GEMMs in the backward: logits recompute, grad_x, grad_w.
    bwd_tflops: float = 0.0
    # "useful" convention: 3 GEMMs total (fwd + grad_x + grad_w), ignoring the
    # backward's logits recompute. This is the convention behind the 405 TFLOP/s
    # figure quoted for the hero run.
    total_useful_tflops: float = 0.0
    # "issued" convention: all 4 GEMMs the kernel actually runs.
    total_issued_tflops: float = 0.0
    peak_hbm_bytes: int = 0
    # correctness
    loss_max_abs_diff_vs_fp32: float = float("nan")
    loss_mean_abs_diff_vs_fp32: float = float("nan")
    loss_bitwise_equal_vs_baseline: Optional[bool] = None
    loss_max_abs_diff_vs_baseline: float = float("nan")
    gx_max_abs_diff: float = float("nan")
    gx_rel_rms: float = float("nan")
    gw_max_abs_diff: float = float("nan")
    gw_rel_rms: float = float("nan")
    # measured kernel launches for ONE fwd+bwd call (CE only, nothing else in the program)
    fwd_launches: int = 0
    bwd_launches: int = 0
    total_launches: int = 0
    mean_launch_us: float = float("nan")
    # profiler-free launch counts derived from the optimized-HLO call graph
    static_fwd_launches: int = 0
    static_total_launches: int = 0
    static_bwd_launches: int = 0
    static_launch_ops: dict[str, int] = field(default_factory=dict)
    # implied mean device-kernel duration, from static launches and measured wall time
    implied_mean_launch_us: float = float("nan")
    launch_lines: dict[str, Any] = field(default_factory=dict)
    opt_hlo_ops: dict[str, int] = field(default_factory=dict)
    note: str = ""


def _gemm_flops(b: int, h: int, v: int) -> float:
    return 2.0 * b * h * v


def _make_inputs(b: int, h: int, v: int, dtype, seed: int):
    key = jax.random.PRNGKey(seed)
    kx, kw, kl = jax.random.split(key, 3)
    # Scale so logits stay in a sane range at H=6144.
    x = (jax.random.normal(kx, (b, h), dtype=jnp.float32) * (1.0 / np.sqrt(h))).astype(dtype)
    w = (jax.random.normal(kw, (h, v), dtype=jnp.float32) * (1.0 / np.sqrt(h))).astype(dtype)
    labels = jax.random.randint(kl, (b,), 0, v, dtype=jnp.int32)
    return jax.block_until_ready((x, w, labels))


def _build_fns(variant: Variant, labels: jax.Array):
    """Return (forward_only_fn, value_and_grad_fn)."""
    block_sizes = BlockSizes(b_block_size=variant.b_block_size, v_block_size=variant.v_block_size)

    def call(x, w):
        if variant.impl == "batched_xla":
            return linear_softmax_cross_entropy_loss_batched_xla(
                x, labels, w, block_sizes=block_sizes, dtype=jnp.float32, precision=None
            )
        return linear_softmax_cross_entropy_loss_xla(
            x,
            labels,
            w,
            block_sizes=block_sizes,
            dtype=jnp.float32,
            precision=None,
            fast_backward=variant.fast_backward,
            bwd_batch_block_size=variant.bwd_batch_block_size,
            bwd_v_block_size=variant.bwd_v_block_size,
        )

    def forward(x, w):
        loss, lse = call(x, w)
        return loss, lse

    def objective(x, w):
        loss, _ = call(x, w)
        # reduction="mean" with unit weights, as in grug loss.py.
        return jnp.mean(loss)

    return jax.jit(forward), jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))


def _time(fn, x, w, steps: int, warmup: int) -> tuple[float, float, Any]:
    start = time.perf_counter()
    out = jax.block_until_ready(fn(x, w))
    compile_s = time.perf_counter() - start
    for _ in range(warmup):
        jax.block_until_ready(fn(x, w))
    start = time.perf_counter()
    for _ in range(steps):
        jax.block_until_ready(fn(x, w))
    return compile_s, (time.perf_counter() - start) / steps, out


def _fp32_reference(x, labels, w, chunk: int):
    """Chunked float32 reference for loss, grad_x and grad_w of ``mean(loss)``.

    Everything is float32 with ``Precision.HIGHEST`` so this is a true fp32
    reference, not a TF32 one.
    """
    b, h = x.shape
    v = w.shape[1]
    w32 = w.astype(jnp.float32)
    scale = jnp.float32(1.0 / b)  # d mean(loss) / d loss_i

    @jax.jit
    def chunk_fn(x_chunk, labels_chunk, w32):
        x32 = x_chunk.astype(jnp.float32)
        logits = jax.lax.dot_general(x32, w32, (((1,), (0,)), ((), ())), precision=jax.lax.Precision.HIGHEST)
        lse = jax.nn.logsumexp(logits, axis=-1)
        label_logits = jnp.take_along_axis(logits, labels_chunk[:, None], axis=1).squeeze(-1)
        loss = lse - label_logits
        probs = jnp.exp(logits - lse[:, None])
        onehot = jnp.arange(v, dtype=jnp.int32)[None, :] == labels_chunk[:, None]
        dlogits = (probs - onehot.astype(jnp.float32)) * scale
        gx = jax.lax.dot_general(dlogits, w32, (((1,), (1,)), ((), ())), precision=jax.lax.Precision.HIGHEST)
        gw = jax.lax.dot_general(x32, dlogits, (((0,), (0,)), ((), ())), precision=jax.lax.Precision.HIGHEST)
        return loss, gx, gw

    losses = []
    gxs = []
    gw_acc = jnp.zeros((h, v), dtype=jnp.float32)
    for start in range(0, b, chunk):
        stop = min(start + chunk, b)
        loss_c, gx_c, gw_c = chunk_fn(x[start:stop], labels[start:stop], w32)
        losses.append(np.asarray(loss_c, dtype=np.float64))
        gxs.append(np.asarray(gx_c, dtype=np.float32))
        gw_acc = gw_acc + gw_c
    return np.concatenate(losses), np.concatenate(gxs), np.asarray(gw_acc, dtype=np.float32)


def _diff(actual, expected) -> tuple[float, float]:
    a = np.asarray(actual, dtype=np.float32)
    e = np.asarray(expected, dtype=np.float32)
    d = np.abs(a - e)
    rel = float(
        np.sqrt(np.mean(d.astype(np.float64) ** 2)) / max(float(np.sqrt(np.mean(e.astype(np.float64) ** 2))), 1e-30)
    )
    return float(d.max()), rel


# Opcodes that never launch a device kernel on XLA:GPU.
_FREE_OPCODES = frozenset(
    {
        "parameter",
        "constant",
        "tuple",
        "get-tuple-element",
        "bitcast",
        "while",
        "conditional",
        "after-all",
        "token",
        "add-dependency",
        "opt-barrier",
        "partition-id",
        "replica-id",
        "call",
        "fusion-noop",
        "bitcast-convert",
    }
)
# Result shapes can be tuples containing spaces, so the opcode is "the first
# lowercase identifier immediately followed by ( after the = sign".
_INSTR_RE = re.compile(r"^\s+%?[\w.\-$]+ = .*?\s([a-z][\w-]*)\((.*)$")
_COMP_RE = re.compile(r"^(?:ENTRY )?%?([\w.\-$]+) \(.*?\) -> .*\{\s*$")
_TRIP_RE = re.compile(r'"known_trip_count":\s*\{\s*"n":\s*"?(\d+)"?')


def _parse_optimized_hlo(text: str):
    """Return (computations, entry_name).

    ``computations[name]`` is a list of ``(opcode, called_computations, trip_count)``.
    """
    comps: dict[str, list] = {}
    entry = None
    cur = None
    for raw in text.splitlines():
        m = _COMP_RE.match(raw)
        if m:
            cur = m.group(1)
            comps[cur] = []
            if raw.lstrip().startswith("ENTRY"):
                entry = cur
            continue
        if raw.strip() == "}":
            cur = None
            continue
        if cur is None:
            continue
        m = _INSTR_RE.match(raw)
        if not m:
            continue
        opcode, rest = m.group(1), m.group(2)
        called = re.findall(
            r"(?:body|condition|to_apply|called_computations=\{?|branch_computations=\{)\s*=?\s*%?([\w.\-$]+)", rest
        )
        trip = _TRIP_RE.search(rest)
        comps[cur].append((opcode, called, int(trip.group(1)) if trip else None))
    return comps, entry


def _static_launch_estimate(text: str) -> tuple[int, dict[str, int]]:
    """Kernel launches per call, derived from the optimized-HLO call graph.

    Walks from ENTRY multiplying by each ``while`` loop's ``known_trip_count``.
    ``fusion`` and ``custom-call`` count as one launch each and are not descended
    into. This needs no profiler, which matters because CUPTI is not usable in
    every container.
    """
    comps, entry = _parse_optimized_hlo(text)
    if entry is None or entry not in comps:
        return 0, {}
    per_opcode: collections.Counter = collections.Counter()
    seen_depth = 0

    def walk(name: str, mult: int, depth: int) -> int:
        nonlocal seen_depth
        seen_depth = max(seen_depth, depth)
        if depth > 12 or name not in comps:
            return 0
        total = 0
        for opcode, called, trip in comps[name]:
            if opcode == "while":
                t = trip if trip is not None else 1
                for sub in called:
                    total += walk(sub, mult * t, depth + 1)
                continue
            if opcode in ("fusion", "custom-call"):
                per_opcode[opcode] += mult
                total += mult
                continue
            if opcode in _FREE_OPCODES:
                continue
            per_opcode[opcode] += mult
            total += mult
        return total

    return walk(entry, 1, 0), dict(per_opcode)


def _static_launches(fn, x, w) -> tuple[int, dict[str, int]]:
    try:
        text = fn.lower(x, w).compile().as_text()
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": str(exc)[:200]}
    return _static_launch_estimate(text)


_KERNEL_LINE_HINTS = ("XLA Ops", "XLA Modules", "Kernels", "Compute")


def _measure_launches(fn, x, w) -> tuple[int, float, dict[str, Any]]:
    """Count device kernel launches for exactly one call of ``fn``.

    Returns (launches, mean_launch_us, per_line_breakdown). The breakdown is kept
    verbatim so the choice of "which xplane line is the kernel line" stays auditable.
    """
    tmpdir = tempfile.mkdtemp(prefix="ce_prof_")
    try:
        with jax.profiler.trace(tmpdir):
            jax.block_until_ready(fn(x, w))
        paths = glob.glob(os.path.join(tmpdir, "**", "*.xplane.pb"), recursive=True)
        if not paths:
            return 0, float("nan"), {"error": "no xplane produced"}
        data = jax.profiler.ProfileData.from_file(paths[0])
        breakdown: dict[str, Any] = {}
        for plane in data.planes:
            for line in plane.lines:
                events = list(line.events)
                if not events:
                    continue
                total_ns = float(sum(e.duration_ns for e in events))
                breakdown[f"{plane.name} :: {line.name}"] = {
                    "n": len(events),
                    "total_ms": total_ns / 1e6,
                    "mean_us": total_ns / len(events) / 1e3,
                }
        # Prefer a device plane's "XLA Ops" line: on GPU that is one event per kernel.
        best_key = None
        for key in breakdown:
            plane_name, _, line_name = key.partition(" :: ")
            if "device" not in plane_name.lower() and "gpu" not in plane_name.lower():
                continue
            if line_name.strip() in _KERNEL_LINE_HINTS[:1]:
                best_key = key
                break
        if best_key is None:
            device_keys = [k for k in breakdown if "device" in k.lower() or "gpu" in k.lower()]
            if device_keys:
                best_key = max(device_keys, key=lambda k: breakdown[k]["n"])
        if best_key is None:
            return 0, float("nan"), breakdown
        chosen = breakdown[best_key]
        breakdown["_chosen_line"] = best_key
        return int(chosen["n"]), float(chosen["mean_us"]), breakdown
    except Exception as exc:  # noqa: BLE001
        return 0, float("nan"), {"error": f"{type(exc).__name__}: {exc}"[:500]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _opt_hlo_ops(fn, x, w) -> dict[str, int]:
    try:
        text = fn.lower(x, w).compile().as_text()
    except Exception:
        return {}
    counts = collections.Counter()
    for m in re.finditer(r"=\s*\S+\s+(fusion|custom-call|while|copy|dot|reduce|transpose|scatter|gather)\b", text):
        counts[m.group(1)] += 1
    return dict(counts)


def run_variant(variant: Variant, args) -> Result:
    res = Result(
        variant=variant.name,
        b=args.batch,
        h=args.hidden,
        v=args.vocab,
        b_block_size=variant.b_block_size,
        v_block_size=variant.v_block_size,
        bwd_v_block_size=variant.bwd_v_block_size,
        bwd_batch_block_size=variant.bwd_batch_block_size,
        note=variant.note,
    )
    device = jax.local_devices()[0]
    res.device_kind = device.device_kind

    dtype = jnp.dtype(args.input_dtype)
    x, w, labels = _make_inputs(args.batch, args.hidden, args.vocab, dtype, args.seed)

    try:
        fwd_fn, grad_fn = _build_fns(variant, labels)
        res.compile_fwd_s, res.fwd_s, _ = _time(fwd_fn, x, w, args.steps, args.warmup)
        res.compile_total_s, res.total_s, (value, (gx, gw)) = _time(grad_fn, x, w, args.steps, args.warmup)
    except Exception as exc:  # noqa: BLE001 - report and keep the sweep going
        res.ok = False
        res.error = f"{type(exc).__name__}: {exc}"[:2000]
        return res

    res.bwd_s = res.total_s - res.fwd_s
    one_gemm = _gemm_flops(args.batch, args.hidden, args.vocab)
    res.fwd_tflops = one_gemm / res.fwd_s / 1e12
    res.bwd_tflops = 3.0 * one_gemm / res.bwd_s / 1e12 if res.bwd_s > 0 else float("nan")
    res.total_useful_tflops = 3.0 * one_gemm / res.total_s / 1e12
    res.total_issued_tflops = 4.0 * one_gemm / res.total_s / 1e12

    stats = device.memory_stats() or {}
    res.peak_hbm_bytes = int(stats.get("peak_bytes_in_use", 0))

    res.static_fwd_launches, _ = _static_launches(fwd_fn, x, w)
    res.static_total_launches, res.static_launch_ops = _static_launches(grad_fn, x, w)
    res.static_bwd_launches = res.static_total_launches - res.static_fwd_launches
    if res.static_total_launches:
        res.implied_mean_launch_us = res.total_s / res.static_total_launches * 1e6

    if args.profile:
        res.fwd_launches, _, _ = _measure_launches(fwd_fn, x, w)
        res.total_launches, res.mean_launch_us, res.launch_lines = _measure_launches(grad_fn, x, w)
        res.bwd_launches = res.total_launches - res.fwd_launches

    if args.hlo_ops:
        res.opt_hlo_ops = _opt_hlo_ops(grad_fn, x, w)

    if args.check:
        loss, _ = jax.block_until_ready(fwd_fn(x, w))
        loss_np = np.asarray(loss, dtype=np.float64)

        # Bitwise comparison of the forward against the current production path.
        base_fwd, _ = _build_fns(VARIANTS_BY_NAME["baseline"], labels)
        base_loss, _ = jax.block_until_ready(base_fwd(x, w))
        base_np = np.asarray(base_loss, dtype=np.float64)
        res.loss_bitwise_equal_vs_baseline = bool(np.array_equal(loss_np, base_np))
        res.loss_max_abs_diff_vs_baseline = float(np.abs(loss_np - base_np).max())
        del base_fwd, base_loss

        ref_loss, ref_gx, ref_gw = _fp32_reference(x, labels, w, args.check_chunk)
        d = np.abs(loss_np - ref_loss)
        res.loss_max_abs_diff_vs_fp32 = float(d.max())
        res.loss_mean_abs_diff_vs_fp32 = float(d.mean())
        res.gx_max_abs_diff, res.gx_rel_rms = _diff(gx, ref_gx)
        res.gw_max_abs_diff, res.gw_rel_rms = _diff(gw, ref_gw)

    return res


def _fanout(args) -> int:
    # NOTE: never touch jax.devices() here. The parent must not initialise a CUDA
    # context, or it holds every GPU and the children fail to allocate.
    names = args.variants
    n_gpu = args.num_gpus
    if n_gpu <= 0:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        n_gpu = len([d for d in visible.split(",") if d.strip()]) if visible else 4
    n_gpu = max(1, n_gpu)
    print(f"fanout: {len(names)} variants over {n_gpu} local GPU(s)", flush=True)
    script = os.path.abspath(__file__)
    base = [
        sys.executable,
        "-u",
        script,
        "--batch",
        str(args.batch),
        "--hidden",
        str(args.hidden),
        "--vocab",
        str(args.vocab),
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
        "--check-chunk",
        str(args.check_chunk),
        "--input-dtype",
        args.input_dtype,
        "--seed",
        str(args.seed),
    ]
    if args.check:
        base.append("--check")
    if args.hlo_ops:
        base.append("--hlo-ops")
    if not args.profile:
        base.append("--no-profile")

    results: list[dict] = []
    for i in range(0, len(names), n_gpu):
        wave = names[i : i + n_gpu]
        procs = []
        for slot, name in enumerate(wave):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
            # Concurrent children must not each grab 75% of a GPU up front; without
            # this the children die with SIGSEGV during the first large allocation.
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
            cmd = base + ["--variants", name]
            print(f"  launching {name} on local GPU {slot}", flush=True)
            procs.append(
                (name, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
            )
        for name, proc in procs:
            try:
                out, _ = proc.communicate(timeout=args.child_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate()
                out = (out or "") + f"\n!! timed out after {args.child_timeout}s"

            got = False
            for line in out.splitlines():
                if line.startswith(RESULT_PREFIX):
                    results.append(json.loads(line[len(RESULT_PREFIX) :]))
                    got = True
            if not got:
                print(f"!! {name} produced no result (exit {proc.returncode}); tail:", flush=True)
                print("\n".join(out.splitlines()[-40:]), flush=True)
                results.append(asdict(Result(variant=name, ok=False, error=f"exit {proc.returncode}")))
    _print_table(results)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)
    return 0


def _print_table(results: list[dict]) -> None:
    print("\n" + "=" * 175, flush=True)
    hdr = (
        f"{'variant':>18} {'fwd_ms':>9} {'bwd_ms':>9} {'total_ms':>9} "
        f"{'useful_TF/s':>12} {'issued_TF/s':>12} {'peak_GiB':>9} "
        f"{'launches':>10} {'us/launch':>10} "
        f"{'loss==base':>11} {'gx_relRMS':>11} {'gw_relRMS':>11}"
    )
    print(hdr)
    print("-" * 175)
    base = next((r for r in results if r["variant"] == "baseline" and r["ok"]), None)
    for r in results:
        if not r["ok"]:
            print(f"{r['variant']:>18}  FAILED: {r['error'][:110]}")
            continue
        print(
            f"{r['variant']:>18} {r['fwd_s'] * 1e3:>9.2f} {r['bwd_s'] * 1e3:>9.2f} {r['total_s'] * 1e3:>9.2f} "
            f"{r['total_useful_tflops']:>12.1f} {r['total_issued_tflops']:>12.1f} "
            f"{r['peak_hbm_bytes'] / 2**30:>9.2f} "
            f"{r.get('static_total_launches', 0):>10} {r.get('implied_mean_launch_us', float('nan')):>10.2f} "
            f"{str(r['loss_bitwise_equal_vs_baseline']):>11} "
            f"{r['gx_rel_rms']:>11.3e} {r['gw_rel_rms']:>11.3e}"
        )
    if base:
        print("-" * 175)
        for r in results:
            if r["ok"] and r["variant"] != "baseline":
                print(
                    f"{r['variant']:>18}  speedup vs baseline: total {base['total_s'] / r['total_s']:.3f}x   "
                    f"bwd {base['bwd_s'] / r['bwd_s']:.3f}x   "
                    f"delta_ms {(r['total_s'] - base['total_s']) * 1e3:+.2f}"
                )
    print("=" * 175, flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=HERO_B)
    p.add_argument("--hidden", type=int, default=HERO_H)
    p.add_argument("--vocab", type=int, default=HERO_V)
    p.add_argument("--input-dtype", type=str, default="bfloat16")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", type=str, default="all", help="comma-separated variant names, or 'all'")
    p.add_argument("--check", action="store_true", help="run the float32 correctness gate")
    p.add_argument("--no-check", action="store_false", dest="check")
    p.set_defaults(check=True)
    p.add_argument("--check-chunk", type=int, default=4096, help="row chunk for the fp32 reference")
    p.add_argument("--hlo-ops", action="store_true", default=False, help="histogram optimized-HLO ops")
    p.add_argument("--profile", action="store_true", help="count device kernel launches via the JAX profiler")
    p.add_argument("--no-profile", action="store_false", dest="profile")
    p.set_defaults(profile=False)
    p.add_argument("--fanout", action="store_true", help="one subprocess per variant across local GPUs")
    p.add_argument(
        "--num-gpus", type=int, default=0, help="GPUs to fan out over (0 = autodetect without touching JAX)"
    )
    p.add_argument("--child-timeout", type=int, default=3600, help="per-variant subprocess timeout, seconds")
    p.add_argument("--out", type=str, default="ce_hero_shape_results.json")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        for v in VARIANTS:
            print(f"{v.name:>18}  {v.note or ''}")
        return 0

    if args.variants == "all":
        names = [v.name for v in VARIANTS]
    else:
        names = [n.strip() for n in args.variants.split(",") if n.strip()]
    unknown = [n for n in names if n not in VARIANTS_BY_NAME]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; known: {sorted(VARIANTS_BY_NAME)}")
    args.variants = names

    if args.fanout:
        return _fanout(args)

    print("devices:", jax.devices(), flush=True)
    print(f"shape: B={args.batch} H={args.hidden} V={args.vocab} dtype={args.input_dtype}", flush=True)
    results = []
    for name in names:
        res = run_variant(VARIANTS_BY_NAME[name], args)
        d = asdict(res)
        results.append(d)
        print(RESULT_PREFIX + json.dumps(d), flush=True)
    if len(results) > 1:
        _print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
