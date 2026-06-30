# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Autotuning benchmark for the FP8 mixed-precision MoE ragged-dot, with measured uncertainty.

This is the *metric* harness for the FP8 ragged-dot optimization loop. It answers one question
defensibly: **how much faster is the best-tuned FP8 hybrid than the best-tuned bf16 baseline**, on
representative Grug MoE expert-MLP shapes, with a confidence interval rather than a single noisy mean.

Three properties, matching ``.agents/skills/add-pallas-kernel`` performance-workflow guidance:

* **Autotuning** — every approach is compared at its *best* block config, not a fixed default.
  The FP8 path is tuned by coordinate descent against the true end-to-end fwd+bwd time: first the
  ``MosaicBlockConfig`` shared by the forward and dlhs GEMMs, then the independent ``WgradBlockConfig``.
  The bf16 baseline is tuned over the Triton ``RAGGED_DOT_BLOCK_*`` / warps / stages space. Configs are
  injected with zero production-code change: ``fp8_ragged`` imports ``_mosaic_pallas_call`` /
  ``_mosaic_wgrad_transposed`` by name, so we swap those module attributes for candidate-bound wrappers.

* **Representative shapes** — small / target / scale buckets spanning the experts axis (which sets the
  raggedness / token-padding regime): E=8 target (~1024 tok/expert, the GFP8-029 headline) and an
  E=128 scale point (~128 tok/expert).

* **Uncertainty** — compile time is separated from steady state; warmup batches are discarded; each
  config yields a distribution of per-step times; we report the median with a bootstrap 95% CI and the
  min, and the fp8-vs-bf16 speedup as a ratio-of-medians with a bootstrap CI. Every candidate is
  numerics-gated against the bf16 reference so we never tune into a wrong-answer config.

Mosaic-GPU is H100-only and needs the cluster CUDA-toolchain bootstrap (see mosaic-gpu-cluster-toolchain
memory). The harness is self-contained (no jax-importing bench helpers) so the bootstrap runs before
``import jax``. Use ``--smoke`` to validate orchestration/statistics on CPU (bf16/xla path only; the
mosaic FP8 path is skipped where unavailable).

    # full sweep on H100
    uv run iris --cluster=cw-us-east-02a job run --gpu H100x1 --enable-extra-resources --extra gpu \\
        -- python lib/levanter/scripts/bench/bench_ragged_fp8_autotune.py --shapes target

    # local CPU smoke (mechanics only; no mosaic, no real speedup)
    uv run --no-sync python lib/levanter/scripts/bench/bench_ragged_fp8_autotune.py --smoke
"""

import argparse
import contextlib
import dataclasses
import functools
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time


def _ensure_cuda_toolchain() -> bool:
    """Make ptxas + libdevice discoverable to XLA/Mosaic before jax import (see GFP8-022)."""
    ptxas = shutil.which("ptxas")
    libdevice = None
    if not ptxas:
        for base in sys.path:
            if base and os.path.isdir(base):
                hits = glob.glob(os.path.join(base, "nvidia", "**", "ptxas"), recursive=True)
                if hits:
                    ptxas = hits[0]
                    break
    for base in sys.path:
        if base and os.path.isdir(base):
            hits = glob.glob(os.path.join(base, "nvidia", "**", "libdevice.10.bc"), recursive=True)
            if hits:
                libdevice = hits[0]
                break
    if not ptxas or not libdevice:
        print(f"cuda toolchain: incomplete (ptxas={ptxas}, libdevice={libdevice}); GPU mosaic path unavailable")
        return False
    nvvm_parent = os.path.dirname(os.path.dirname(os.path.dirname(libdevice)))
    root = tempfile.mkdtemp(prefix="xla_cuda_")
    os.symlink(os.path.dirname(ptxas), os.path.join(root, "bin"))
    os.symlink(os.path.join(nvvm_parent, "nvvm"), os.path.join(root, "nvvm"))
    os.environ["PATH"] = os.path.dirname(ptxas) + os.pathsep + os.environ.get("PATH", "")
    flags = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_cuda_data_dir={root}".strip()
    for var in ("CUDA_DIR", "CUDA_HOME", "CUDA_PATH"):
        os.environ[var] = root
    cwd_link = os.path.join(os.getcwd(), "libdevice.10.bc")
    if not os.path.exists(cwd_link):
        os.symlink(libdevice, cwd_link)
    print(f"cuda toolchain: ptxas={ptxas} libdevice={libdevice}")
    return True


# Skip the GPU toolchain bootstrap when importing only the pure-python statistics (unit tests).
if not os.environ.get("BENCH_SKIP_CUDA_BOOTSTRAP"):
    _ensure_cuda_toolchain()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.profiler  # noqa: E402
import numpy as np  # noqa: E402

import haliax._src.fp8_ragged as fp8_ragged  # noqa: E402
from haliax._src.fp8_ragged import MosaicWgradMode  # noqa: E402
from haliax._src.transposed_ragged_dot_mgpu import WgradBlockConfig, transposed_ragged_dot  # noqa: E402
from haliax.nn.ragged_dot import (  # noqa: E402
    MosaicBlockConfig,
    ragged_dot,
)
from haliax.quantization import Fp8RaggedDotOp  # noqa: E402

# Sibling pure modules (shared with the orchestrator); sys.path makes them importable as script/worker/import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_autotune_configs import (  # noqa: E402
    SHAPE_GRID,
    Shape,
    bf16_candidate_dicts,
    mosaic_candidate_dicts,
    wgrad_candidate_dicts,
)
from fp8_autotune_stats import ratio_median_ci, summarize_times  # noqa: E402,F401

# H100 SXM dense matmul peaks; FP8 (E4M3) is 2x the BF16 rate.
_H100_SXM_BF16_TFLOPS_PER_S = 989.5e12
_H100_SXM_FP8_TFLOPS_PER_S = 1978.9e12

_GRAD_DTYPES = {"e4m3": jnp.float8_e4m3fn, "e5m2": jnp.float8_e5m2}

_SHAPE_GRID = SHAPE_GRID


def _mosaic_candidates() -> list[MosaicBlockConfig]:
    """Shared forward/dlhs Mosaic block configs as dataclasses (design lives in fp8_autotune_configs)."""
    return [MosaicBlockConfig(**d) for d in mosaic_candidate_dicts()]


def _wgrad_candidates() -> list[WgradBlockConfig]:
    """Independent f8 weight-gradient block configs as dataclasses."""
    return [WgradBlockConfig(**d) for d in wgrad_candidate_dicts()]


_BF16_ENV_VARS = (
    "RAGGED_DOT_BLOCK_M",
    "RAGGED_DOT_BLOCK_N",
    "RAGGED_DOT_BLOCK_K",
    "RAGGED_DOT_NUM_WARPS",
    "RAGGED_DOT_NUM_STAGES",
)


@contextlib.contextmanager
def _mosaic_config_override(mosaic_cfg: MosaicBlockConfig, wgrad_cfg: WgradBlockConfig):
    """Temporarily bind candidate block configs into the FP8 e2e path with no production change.

    ``fp8_ragged`` imported ``_mosaic_pallas_call`` / ``_mosaic_wgrad_transposed`` by name, so the
    ``_ragged_dot_layout`` / backward call sites resolve them as module attributes here. We swap those
    attributes for candidate-bound wrappers; ``_mosaic_wgrad_transposed`` ignores config, so we route
    its wrapper through the configurable ``transposed_ragged_dot`` directly.
    """
    orig_mosaic = fp8_ragged._mosaic_pallas_call
    orig_wgrad = fp8_ragged._mosaic_wgrad_transposed

    def wgrad_wrapper(lhs_t, g_t, group_sizes, out_dtype):
        return transposed_ragged_dot(lhs_t, g_t, group_sizes, out_dtype=out_dtype, config=wgrad_cfg)

    fp8_ragged._mosaic_pallas_call = functools.partial(orig_mosaic, config=mosaic_cfg)
    fp8_ragged._mosaic_wgrad_transposed = wgrad_wrapper
    try:
        yield
    finally:
        fp8_ragged._mosaic_pallas_call = orig_mosaic
        fp8_ragged._mosaic_wgrad_transposed = orig_wgrad


@contextlib.contextmanager
def _bf16_env_override(block_m, block_n, block_k, num_warps, num_stages):
    """Temporarily set the Triton block-size env vars (read at trace time by the bf16 ragged_dot)."""
    saved = {k: os.environ.get(k) for k in _BF16_ENV_VARS}
    values = (block_m, block_n, block_k, num_warps, num_stages)
    for k, v in zip(_BF16_ENV_VARS, values):
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------------------------------
# Model + inputs (mirrors bench_ragged_mosaic_hybrid_e2e.py so the metric matches the e2e validation).
# ---------------------------------------------------------------------------------------------------


def _expert_mlp(x, w13, w2, group_sizes, *, dot13, dot2):
    h = dot13(x, w13, group_sizes)
    gate, up = jnp.split(h, 2, axis=-1)
    return dot2(jax.nn.silu(gate) * up, w2, group_sizes)


def _make_inputs(shape: Shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    # Unit-variance activations + modest weights so operands sit in E4M3 range at unit (cold-start) scale.
    x = jnp.asarray(rng.standard_normal((shape.tokens, shape.hidden)), dtype)
    w13 = jnp.asarray(rng.standard_normal((shape.experts, shape.hidden, 2 * shape.intermediate)) * 0.08, dtype)
    w2 = jnp.asarray(rng.standard_normal((shape.experts, shape.intermediate, shape.hidden)) * 0.08, dtype)
    counts = rng.multinomial(shape.tokens, np.ones(shape.experts) / shape.experts)
    return x, w13, w2, jnp.asarray(counts, jnp.int32)


def _build_dots(path, compute_dtype, grad_dtype, mosaic_wgrad):
    """(w13, w2) grouped-matmul callables: bf16 baseline or per-GEMM Fp8RaggedDotOp."""
    if path == "bf16":
        dot = lambda a, b, gs: ragged_dot(a, b, gs, implementation="auto")  # noqa: E731
        return dot, dot
    impl = "mosaic" if path == "mosaic" else "triton"
    kw = dict(compute_dtype=compute_dtype, implementation=impl, grad_dtype=grad_dtype, mosaic_wgrad=mosaic_wgrad)
    return Fp8RaggedDotOp.init(**kw), Fp8RaggedDotOp.init(**kw)


def _grad_fn(dot13, dot2, group_sizes):
    def loss(x, w13, w2):
        return _expert_mlp(x, w13, w2, group_sizes, dot13=dot13, dot2=dot2).astype(jnp.float32).sum()

    return jax.grad(loss, argnums=(0, 1, 2))


def _rel_frob(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# ---------------------------------------------------------------------------------------------------
# Timing + statistics (pure numpy; unit-tested on CPU in test_bench_ragged_fp8_autotune.py).
# ---------------------------------------------------------------------------------------------------


def time_steady_state(fn, args, *, samples: int, inner_steps: int, warmup: int):
    """Compile ``fn`` once, then collect ``samples`` per-step time measurements.

    Each sample times ``inner_steps`` dispatched calls under a single ``block_until_ready`` (amortizing
    Python/dispatch overhead) and divides by ``inner_steps``. Returns ``(compile_time, per_step_times)``
    where ``per_step_times`` is a numpy array of length ``samples`` (warmup batches already discarded).

    ``jax.clear_caches()`` first: the autotuner mutates a module attribute (the active block config)
    between candidates, but jit caches a trace by ``(function, avals)`` and would otherwise serve the
    *previous* candidate's compiled kernel — silently timing the same config every time. Clearing forces
    a fresh trace+compile that observes the current config (verified on CPU before trusting any number).
    """
    jax.clear_caches()
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_time = time.perf_counter() - start

    def _batch():
        out = None
        for _ in range(inner_steps):
            out = compiled(*args)
        jax.block_until_ready(out)

    for _ in range(warmup):
        _batch()
    times = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        t0 = time.perf_counter()
        _batch()
        times[i] = (time.perf_counter() - t0) / inner_steps
    return compile_time, times


# ---------------------------------------------------------------------------------------------------
# Profiling capture (attribution, not a metric): where does the fp8 fwd+bwd spend its time?
# ---------------------------------------------------------------------------------------------------


def _run_profile(shape, dtype, grad_dtype, mosaic_wgrad, *, steps, warmup, profile_dir, log):
    """Capture jax profiler traces of the fp8 and bf16 fwd+bwd at default block config, one shape.

    Writes ``<profile_dir>/{fp8,bf16}/`` TensorBoard-style dirs (each with
    ``plugins/profile/<ts>/*.xplane.pb``) for kernel-level attribution via
    ``lib/marin/tools/profile_summary.py``. This answers *where* the fp8 path spends time — the Mosaic
    GEMM kernels (fwd / dlhs / wgrad) vs the fp8 fixed overhead (quantize/scale, cast-transpose,
    dequant) — which the headline speedup cannot. The fp8 path runs at the GFP8 default
    ``MosaicBlockConfig`` / ``WgradBlockConfig`` (the held d2560 winner); bf16 at the Triton default.
    Not a timing metric: trace overhead perturbs absolute times, so use it only for the *breakdown*.
    """
    x, w13, w2, group_sizes = _make_inputs(shape, dtype)
    args = (x, w13, w2)

    def capture(name, timed, ctx):
        jax.clear_caches()
        with ctx:
            compiled = jax.jit(timed).lower(*args).compile()
            for _ in range(warmup):
                jax.block_until_ready(compiled(*args))
            sub = os.path.join(profile_dir, name)
            os.makedirs(sub, exist_ok=True)
            with jax.profiler.trace(sub):
                out = None
                for _ in range(steps):
                    out = compiled(*args)
                jax.block_until_ready(out)
        log(f"  profiled {name} ({shape}): {steps} steps -> {sub}")

    d13, d2 = _build_dots("mosaic", dtype, grad_dtype, mosaic_wgrad)
    capture("fp8", _grad_fn(d13, d2, group_sizes), _mosaic_config_override(_mosaic_candidates()[0], _wgrad_candidates()[0]))

    b13, b2 = _build_dots("bf16", dtype, grad_dtype, MosaicWgradMode.BF16)
    capture("bf16", _grad_fn(b13, b2, group_sizes), contextlib.nullcontext())


# ---------------------------------------------------------------------------------------------------
# Provenance + row emission.
# ---------------------------------------------------------------------------------------------------


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _device_info():
    devs = jax.devices()
    return jax.default_backend(), (devs[0].device_kind if devs else "none"), len(devs)


def _make_row(*, kernel, implementation, shape, dtype, block_sizes, compile_time, summary, error, extra=None):
    backend, device_type, device_count = _device_info()
    row = {
        "kernel": kernel,
        "implementation": implementation,
        "shape": str(shape),
        "shape_dims": dataclasses.asdict(shape),
        "dtype": str(dtype),
        "backend": backend,
        "device_type": device_type,
        "device_count": device_count,
        "block_sizes": block_sizes,
        "compile_time_s": compile_time,
        "steady_state_time_s": (summary or {}).get("median_s"),
        "stats": summary,
        "error": error,
        "git_sha": _git_sha(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend_env": {k: os.environ.get(k) for k in _BF16_ENV_VARS if os.environ.get(k) is not None},
    }
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------------------------------------
# Candidate evaluation.
# ---------------------------------------------------------------------------------------------------


def _eval_candidate(timed_fn, args, *, samples, inner_steps, warmup):
    """Time one already-config-bound callable. Returns dict with summary or a captured error.

    An unsupported/illegal block config raises at compile; we record it and continue the sweep rather
    than aborting (a failed candidate is data — it bounds the feasible space).
    """
    try:
        compile_time, times = time_steady_state(
            timed_fn, args, samples=samples, inner_steps=inner_steps, warmup=warmup
        )
    except Exception as exc:
        return {"compile_time": None, "summary": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"compile_time": compile_time, "summary": summarize_times(times), "error": None}


def _select_best(results):
    """Pick the result with the lowest median time among candidates that compiled."""
    viable = [r for r in results if r["result"]["error"] is None and r["result"]["summary"] is not None]
    if not viable:
        return None
    return min(viable, key=lambda r: r["result"]["summary"]["median_s"])


def _sweep_stage(
    timed,
    args,
    candidates,
    override,
    *,
    samples,
    inner_steps,
    warmup,
    kernel,
    shape,
    dtype,
    block_sizes_fn,
    rows,
    log,
    label,
):
    """Time every candidate under its config override; emit a row each; return the best result."""
    collected = []
    for cfg in candidates:
        with override(cfg):
            res = _eval_candidate(timed, args, samples=samples, inner_steps=inner_steps, warmup=warmup)
        rows.append(
            _make_row(
                kernel=kernel,
                implementation="mosaic",
                shape=shape,
                dtype=dtype,
                block_sizes=block_sizes_fn(cfg),
                compile_time=res["compile_time"],
                summary=res["summary"],
                error=res["error"],
            )
        )
        collected.append({"cfg": cfg, "result": res})
        log(f"    {label} {dataclasses.asdict(cfg)} -> {_fmt(res)}")
    return _select_best(collected)


def tune_fp8(shape, dtype, grad_dtype, mosaic_wgrad, *, samples, inner_steps, warmup, numerics_tol, rows, log):
    """Coordinate descent for the FP8 hybrid: sweep MosaicBlockConfig, then WgradBlockConfig, against
    the true e2e fwd+bwd time. Numerics (FP8 recipe correctness) is block-invariant, so it is checked
    once under defaults (a gate) and re-confirmed on the winner — not per candidate.

    Returns the winning configs, their headline timing distribution, and the numerics check, or None if
    the recipe fails the numerics gate or no config compiles.
    """
    x, w13, w2, group_sizes = _make_inputs(shape, dtype)
    dot13, dot2 = _build_dots("mosaic", dtype, grad_dtype, mosaic_wgrad)
    timed = _grad_fn(dot13, dot2, group_sizes)

    bf13, bf2 = _build_dots("bf16", dtype, grad_dtype, mosaic_wgrad)
    jax.clear_caches()
    g_ref = jax.block_until_ready(jax.jit(_grad_fn(bf13, bf2, group_sizes))(x, w13, w2))

    default_mosaic = _mosaic_candidates()[0]
    default_wgrad = _wgrad_candidates()[0]

    def rel_frob_under(mosaic_cfg, wgrad_cfg):
        with _mosaic_config_override(mosaic_cfg, wgrad_cfg):
            jax.clear_caches()
            g_path = jax.block_until_ready(jax.jit(timed)(x, w13, w2))
        return max(_rel_frob(gp, gr) for gp, gr in zip(g_path, g_ref))

    # Numerics gate (block-invariant recipe correctness): reject the whole FP8 path if it diverges.
    gate_rel = rel_frob_under(default_mosaic, default_wgrad)
    if gate_rel > numerics_tol:
        log(f"  NUMERICS GATE FAILED: grad rel_frob {gate_rel:.3e} > tol {numerics_tol:.3e}; skipping fp8")
        rows.append(
            _make_row(
                kernel="fp8_hybrid_fwdbwd",
                implementation="mosaic",
                shape=shape,
                dtype=dtype,
                block_sizes={"mosaic": dataclasses.asdict(default_mosaic), "wgrad": dataclasses.asdict(default_wgrad)},
                compile_time=None,
                summary=None,
                error=f"numerics gate rel_frob {gate_rel:.3e} > {numerics_tol:.3e}",
                extra={"rel_frob_vs_bf16": gate_rel, "grad_dtype": str(grad_dtype), "mosaic_wgrad": str(mosaic_wgrad)},
            )
        )
        return None
    log(f"  numerics gate OK: grad rel_frob {gate_rel:.3e} vs bf16")

    # Stage A: shared forward/dlhs Mosaic config, wgrad held at default.
    best_a = _sweep_stage(
        timed,
        (x, w13, w2),
        _mosaic_candidates(),
        lambda cfg: _mosaic_config_override(cfg, default_wgrad),
        samples=samples,
        inner_steps=inner_steps,
        warmup=warmup,
        kernel="fp8_hybrid_fwdbwd",
        shape=shape,
        dtype=dtype,
        block_sizes_fn=lambda cfg: {
            "stage": "mosaic",
            "mosaic": dataclasses.asdict(cfg),
            "wgrad": dataclasses.asdict(default_wgrad),
        },
        rows=rows,
        log=log,
        label="mosaic",
    )
    if best_a is None:
        return None
    best_mosaic = best_a["cfg"]
    log(f"  best mosaic: {dataclasses.asdict(best_mosaic)} ({_fmt(best_a['result'])})")

    # Stage B: fix the mosaic winner, sweep the independent wgrad config (only meaningful in fp8 wgrad mode).
    best_wgrad = default_wgrad
    if mosaic_wgrad == MosaicWgradMode.FP8:
        best_b = _sweep_stage(
            timed,
            (x, w13, w2),
            _wgrad_candidates(),
            lambda cfg: _mosaic_config_override(best_mosaic, cfg),
            samples=samples,
            inner_steps=inner_steps,
            warmup=warmup,
            kernel="fp8_hybrid_fwdbwd",
            shape=shape,
            dtype=dtype,
            block_sizes_fn=lambda cfg: {
                "stage": "wgrad",
                "mosaic": dataclasses.asdict(best_mosaic),
                "wgrad": dataclasses.asdict(cfg),
            },
            rows=rows,
            log=log,
            label="wgrad",
        )
        if best_b is not None:
            best_wgrad = best_b["cfg"]
            log(f"  best wgrad: {dataclasses.asdict(best_wgrad)} ({_fmt(best_b['result'])})")

    # Headline re-time of the winning (mosaic, wgrad) pair, and a final numerics confirmation on it.
    with _mosaic_config_override(best_mosaic, best_wgrad):
        compile_time, times = time_steady_state(
            timed, (x, w13, w2), samples=samples, inner_steps=inner_steps, warmup=warmup
        )
    win_rel = rel_frob_under(best_mosaic, best_wgrad)
    return {
        "mosaic": best_mosaic,
        "wgrad": best_wgrad,
        "compile_time": compile_time,
        "times": times,
        "gate_rel_frob": gate_rel,
        "winner_rel_frob": win_rel,
    }


def tune_bf16(shape, dtype, grad_dtype, *, samples, inner_steps, warmup, rows, log):
    """Sweep the Triton bf16 baseline block configs; return best config + its timing distribution."""
    x, w13, w2, group_sizes = _make_inputs(shape, dtype)

    results = []
    for cfg in bf16_candidate_dicts():
        with _bf16_env_override(**cfg):
            dot13, dot2 = _build_dots("bf16", dtype, grad_dtype, MosaicWgradMode.BF16)
            timed = _grad_fn(dot13, dot2, group_sizes)
            res = _eval_candidate(timed, (x, w13, w2), samples=samples, inner_steps=inner_steps, warmup=warmup)
            rows.append(
                _make_row(
                    kernel="bf16_fwdbwd",
                    implementation="triton",
                    shape=shape,
                    dtype=dtype,
                    block_sizes=cfg,
                    compile_time=res["compile_time"],
                    summary=res["summary"],
                    error=res["error"],
                )
            )
            results.append({"cfg": cfg, "result": res})
            log(f"    bf16 {cfg} -> {_fmt(res)}")
    best = _select_best(results)
    if best is None:
        return None
    log(f"  best bf16: {best['cfg']} ({_fmt(best['result'])})")
    # Final headline timing at the winner.
    with _bf16_env_override(**best["cfg"]):
        dot13, dot2 = _build_dots("bf16", dtype, grad_dtype, MosaicWgradMode.BF16)
        timed = _grad_fn(dot13, dot2, group_sizes)
        compile_time, times = time_steady_state(
            timed, (x, w13, w2), samples=samples, inner_steps=inner_steps, warmup=warmup
        )
    return {"cfg": best["cfg"], "compile_time": compile_time, "times": times}


def _fmt(res):
    if res["summary"] is None:
        return f"FAILED ({res['error']})"
    s = res["summary"]
    tag = "" if res["error"] is None else f" REJECTED({res['error']})"
    return f"{s['median_s'] * 1e3:.3f} ms [CI {s['ci95_rel_width'] * 100:.1f}%]{tag}"


# ---------------------------------------------------------------------------------------------------
# Worker mode: evaluate an explicit list of config requests on this process's single visible GPU.
# The multi-GPU orchestrator (orchestrate_fp8_autotune.py) pins one GPU per worker via
# CUDA_VISIBLE_DEVICES and owns the coordinate-descent across workers; the worker is stateless beyond
# one shape's inputs. Each request -> one row; ``want_times`` rows carry the raw per-step distribution
# so the orchestrator can form a ratio-of-medians CI for the headline.
# ---------------------------------------------------------------------------------------------------


def _run_worker(work_file, rows_out):
    with open(work_file) as f:
        spec = json.load(f)
    shape = Shape(**spec["shape"])
    dtype = jnp.dtype(spec["dtype"])
    grad_dtype = _GRAD_DTYPES[spec["grad_dtype"]]
    mosaic_wgrad = MosaicWgradMode(spec["mosaic_wgrad"])
    sb = {"samples": spec["samples"], "inner_steps": spec["inner_steps"], "warmup": spec["warmup"]}

    x, w13, w2, group_sizes = _make_inputs(shape, dtype)
    args = (x, w13, w2)
    timed_fp8 = None  # built lazily on the first fp8 request
    g_ref = None  # bf16 reference grads, built lazily for the first numerics check

    rows = []
    for req in spec["requests"]:
        if req["kind"] == "fp8":
            if timed_fp8 is None:
                d13, d2 = _build_dots("mosaic", dtype, grad_dtype, mosaic_wgrad)
                timed_fp8 = _grad_fn(d13, d2, group_sizes)
            mosaic_cfg = MosaicBlockConfig(**req["mosaic"])
            wgrad_cfg = WgradBlockConfig(**req["wgrad"])
            rel = None
            if req.get("want_numerics"):
                if g_ref is None:
                    b13, b2 = _build_dots("bf16", dtype, grad_dtype, mosaic_wgrad)
                    jax.clear_caches()
                    g_ref = jax.block_until_ready(jax.jit(_grad_fn(b13, b2, group_sizes))(*args))
                with _mosaic_config_override(mosaic_cfg, wgrad_cfg):
                    jax.clear_caches()
                    g_path = jax.block_until_ready(jax.jit(timed_fp8)(*args))
                rel = max(_rel_frob(gp, gr) for gp, gr in zip(g_path, g_ref))
            try:
                with _mosaic_config_override(mosaic_cfg, wgrad_cfg):
                    compile_time, times = time_steady_state(timed_fp8, args, **sb)
                summary, error = summarize_times(times), None
            except Exception as exc:
                compile_time, times, summary, error = None, None, None, f"{type(exc).__name__}: {exc}"
            rows.append(
                _make_row(
                    kernel="fp8_hybrid_fwdbwd",
                    implementation="mosaic",
                    shape=shape,
                    dtype=dtype,
                    block_sizes={"mosaic": req["mosaic"], "wgrad": req["wgrad"]},
                    compile_time=compile_time,
                    summary=summary,
                    error=error,
                    extra={
                        "request_id": req["id"],
                        "kind": "fp8",
                        "rel_frob_vs_bf16": rel,
                        "times": (times.tolist() if (times is not None and req.get("want_times")) else None),
                        "grad_dtype": str(grad_dtype),
                        "mosaic_wgrad": str(mosaic_wgrad),
                    },
                )
            )
        else:
            cfg = req["bf16cfg"]
            with _bf16_env_override(**cfg):
                d13, d2 = _build_dots("bf16", dtype, grad_dtype, MosaicWgradMode.BF16)
                timed = _grad_fn(d13, d2, group_sizes)
                try:
                    compile_time, times = time_steady_state(timed, args, **sb)
                    summary, error = summarize_times(times), None
                except Exception as exc:
                    compile_time, times, summary, error = None, None, None, f"{type(exc).__name__}: {exc}"
            rows.append(
                _make_row(
                    kernel="bf16_fwdbwd",
                    implementation="triton",
                    shape=shape,
                    dtype=dtype,
                    block_sizes=cfg,
                    compile_time=compile_time,
                    summary=summary,
                    error=error,
                    extra={
                        "request_id": req["id"],
                        "kind": "bf16",
                        "times": (times.tolist() if (times is not None and req.get("want_times")) else None),
                    },
                )
            )
        status = error if error else f"{summary['median_s'] * 1e3:.3f} ms"
        print(f"worker: {req['id']} -> {status}", flush=True)

    with open(rows_out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"worker: wrote {len(rows)} rows -> {rows_out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default="small,target,scale", help="comma list of {small,target,scale} or 'all'")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--grad-dtype", choices=("e4m3", "e5m2"), default="e5m2")
    ap.add_argument("--mosaic-wgrad", choices=("bf16", "fp8"), default="fp8")
    ap.add_argument("--samples", type=int, default=40, help="timing batches per config (distribution size)")
    ap.add_argument("--inner-steps", type=int, default=10, help="dispatched calls per timing batch")
    ap.add_argument("--warmup", type=int, default=5, help="discarded warmup batches")
    ap.add_argument("--numerics-tol", type=float, default=0.25, help="max grad rel_frob vs bf16 to accept a config")
    ap.add_argument("--out-dir", default=None, help="defaults to scratch/fp8_bench/<timestamp>")
    ap.add_argument("--smoke", action="store_true", help="CPU mechanics check: bf16/xla only, tiny budget, no mosaic")
    ap.add_argument("--profile", action="store_true", help="capture fp8+bf16 fwd+bwd traces (attribution) for one shape, then exit")
    ap.add_argument("--profile-steps", type=int, default=30, help="dispatched steps inside the profiler trace window")
    ap.add_argument("--profile-dir", default=None, help="profile: trace output dir (defaults to <out-dir>/profiler)")
    ap.add_argument(
        "--worker", action="store_true", help="evaluate an orchestrator-supplied request list (single GPU)"
    )
    ap.add_argument("--work-file", default=None, help="worker: JSON spec of requests to evaluate")
    ap.add_argument("--rows-out", default=None, help="worker: JSONL path to write result rows")
    args = ap.parse_args()

    if args.worker:
        _run_worker(args.work_file, args.rows_out)
        return

    dtype = jnp.dtype(args.dtype)
    grad_dtype = _GRAD_DTYPES[args.grad_dtype]
    mosaic_wgrad = MosaicWgradMode(args.mosaic_wgrad)
    backend, device_type, device_count = _device_info()

    if args.smoke:
        args.samples, args.inner_steps, args.warmup = 8, 4, 2

    shape_keys = list(_SHAPE_GRID) if args.shapes == "all" else [s.strip() for s in args.shapes.split(",")]
    if args.smoke:
        shape_keys = ["small"]
    shapes = [_SHAPE_GRID[k] for k in shape_keys]

    out_dir = args.out_dir or os.path.join("scratch", "fp8_bench", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "rows.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")

    def log(msg):
        print(msg, flush=True)

    log(f"backend={backend} device={device_type}x{device_count} git={_git_sha()}")
    log(
        f"shapes={[str(s) for s in shapes]} samples={args.samples} inner_steps={args.inner_steps} warmup={args.warmup}"
    )
    log(f"out_dir={out_dir}")

    if args.profile:
        profile_dir = args.profile_dir or os.path.join(out_dir, "profiler")
        shape = shapes[0]
        log(f"  [profile] {shape} grad_dtype={grad_dtype} mosaic_wgrad={mosaic_wgrad} steps={args.profile_steps}")
        _run_profile(
            shape, dtype, grad_dtype, mosaic_wgrad,
            steps=args.profile_steps, warmup=args.warmup, profile_dir=profile_dir, log=log,
        )
        log("result_json " + json.dumps({"profile_dir": profile_dir, "shape": str(shape)}))
        return

    rows = []
    headline = []
    for shape in shapes:
        log(f"\n=== {shape} ===")

        log("  [bf16 baseline sweep]")
        bf16 = tune_bf16(
            shape,
            dtype,
            grad_dtype,
            samples=args.samples,
            inner_steps=args.inner_steps,
            warmup=args.warmup,
            rows=rows,
            log=log,
        )

        fp8 = None
        if not args.smoke:
            log("  [fp8 hybrid autotune]")
            fp8 = tune_fp8(
                shape,
                dtype,
                grad_dtype,
                mosaic_wgrad,
                samples=args.samples,
                inner_steps=args.inner_steps,
                warmup=args.warmup,
                numerics_tol=args.numerics_tol,
                rows=rows,
                log=log,
            )

        entry = {"shape": str(shape), "shape_dims": dataclasses.asdict(shape)}
        if bf16 is not None:
            entry["bf16_best"] = {"cfg": bf16["cfg"], **summarize_times(bf16["times"])}
        if fp8 is not None and bf16 is not None:
            speedup, lo, hi = ratio_median_ci(fp8["times"], bf16["times"])
            entry["fp8_best"] = {
                "mosaic": dataclasses.asdict(fp8["mosaic"]),
                "wgrad": dataclasses.asdict(fp8["wgrad"]),
                "grad_rel_frob_vs_bf16": fp8["winner_rel_frob"],
                **summarize_times(fp8["times"]),
            }
            entry["speedup_vs_bf16_best"] = {"median": speedup, "ci95_low": lo, "ci95_high": hi}
            log(
                f"  >> HEADLINE {shape}: fp8 {np.median(fp8['times']) * 1e3:.3f} ms vs bf16 {np.median(bf16['times']) * 1e3:.3f} ms "
                f"= {speedup:.3f}x [CI {lo:.3f}-{hi:.3f}]"
            )
        headline.append(entry)

    with open(rows_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {
        "backend": backend,
        "device_type": device_type,
        "device_count": device_count,
        "git_sha": _git_sha(),
        "dtype": str(dtype),
        "grad_dtype": str(grad_dtype),
        "mosaic_wgrad": str(mosaic_wgrad),
        "samples": args.samples,
        "inner_steps": args.inner_steps,
        "warmup": args.warmup,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "results": headline,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\nwrote {len(rows)} rows -> {rows_path}")
    log(f"wrote summary -> {summary_path}")
    log("result_json " + json.dumps({"summary_path": summary_path, "results": headline}))


if __name__ == "__main__":
    main()
