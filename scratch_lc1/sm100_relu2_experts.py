# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GB200 parity + per-kernel timing for the ungated ReLU^2 expert MLP on QuACK SM100 grouped GEMMs.

Compares `_CuteRelu2ExpertMlp` (bf16) against `_RaggedDotExpertMlp(fused_relu2)` run in fp32 on the same
bf16-rounded inputs, at a small and at the hero per-GPU shape, with uneven group sizes (one empty) and
active rows < physical rows (the tail filled with NaN, which no kernel may read). Reports max / median
relative errors on out, dx, dw13, dw2 and output_dot over the active rows, then times the six QuACK GEMMs
of the new path at a few tile/cluster configurations.

Usage: python scratch_lc1/sm100_relu2_experts.py [parity,kernels]
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug._moe import sonic_cute
from levanter.grug._moe.ep_ragged_all_to_all import _CuteRelu2ExpertMlp, _RaggedDotExpertMlp, _select_expert_mlp
from levanter.grug._moe.quack_moe_cute import (
    quack_act_grouped_gemm,
    quack_dact_grouped_gemm,
    quack_grouped_gemm,
    quack_grouped_wgrad,
)
from levanter.kernels.pallas.relu2_mlp import fused_relu2

# The fp32 reference must not run its GEMMs in TF32.
jax.config.update("jax_default_matmul_precision", "highest")


def bench(fn, *args, iters=10, warmup=3):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def rel_errors(got, want):
    got = np.asarray(got, np.float32)
    want = np.asarray(want, np.float32)
    assert np.all(np.isfinite(got)), "non-finite values in active rows"
    scale = float(np.max(np.abs(want))) or 1.0
    err = np.abs(got - want)
    # Elementwise relative error, floored at 1% of the tensor's scale so near-zero entries don't dominate.
    elem = err / np.maximum(np.abs(want), 1e-2 * scale)
    fro = float(np.linalg.norm(err.ravel()) / max(np.linalg.norm(want.ravel()), 1e-30))
    return dict(max_norm=float(err.max() / scale), max_rel=float(elem.max()), median_rel=float(np.median(elem)), fro=fro)


def parity(name, E, H, I, O, physical_sizes, active_sizes, seed=0):
    physical_sizes = np.asarray(physical_sizes, np.int32)
    active_sizes = np.asarray(active_sizes, np.int32)
    assert np.all(active_sizes <= physical_sizes)
    C = int(physical_sizes.sum())
    n_active = int(active_sizes.sum())
    ks = jax.random.split(jax.random.PRNGKey(seed), 4)
    x = (jax.random.normal(ks[0], (C, H), jnp.float32) * 0.5).astype(jnp.bfloat16)
    wu = (jax.random.normal(ks[1], (E, H, I), jnp.float32) * H**-0.5).astype(jnp.bfloat16)
    wd = (jax.random.normal(ks[2], (E, I, O), jnp.float32) * I**-0.5).astype(jnp.bfloat16)
    dy = jax.random.normal(ks[3], (C, O), jnp.float32).astype(jnp.bfloat16)
    # The receiver buffer is expert-major with the active rows a prefix of the buffer: pack the active
    # rows of every expert contiguously, matching how `_CuteExpertMlp`'s callers lay them out.
    tail = jnp.arange(C) >= n_active
    x = jnp.where(tail[:, None], jnp.nan, x).astype(jnp.bfloat16)
    dy = jnp.where(tail[:, None], jnp.nan, dy).astype(jnp.bfloat16)
    phys = jnp.asarray(physical_sizes)
    act = jnp.asarray(active_sizes)

    cute = _CuteRelu2ExpertMlp()
    ref = _RaggedDotExpertMlp(fused_relu2)

    @jax.jit
    def run_cute(x, wu, wd, dy):
        out, res = cute.forward(x, wu, wd, phys, act)
        return out, cute.backward(res, dy)

    @jax.jit
    def run_ref(x, wu, wd, dy):
        f32 = lambda a: a.astype(jnp.float32)  # noqa: E731
        # The reference zero-masks rows past the active count itself, but a NaN times 0 is NaN in its dx
        # mask path, so give it clean zeros there.
        x = jnp.where(tail[:, None], 0, f32(x))
        dy = jnp.where(tail[:, None], 0, f32(dy))
        out, res = ref.forward(x, f32(wu), f32(wd), act.at[-1].add(C - n_active), act)
        return out, ref.backward(res, dy)

    out, (dx, dw13, dw2, dot) = run_cute(x, wu, wd, dy)
    rout, (rdx, rdw13, rdw2, rdot) = run_ref(x, wu, wd, dy)
    a = slice(0, n_active)
    print(
        f"PARITY {name}: E{E} H{H} I{I} O{O} physical {physical_sizes.tolist()} active {active_sizes.tolist()}",
        flush=True,
    )
    for tag, got, want in [
        ("out", out[a], rout[a]),
        ("dx", dx[a], rdx[a]),
        ("dw13", dw13, rdw13),
        ("dw2", dw2, rdw2),
        ("output_dot", dot[a], rdot[a]),
    ]:
        e = rel_errors(got, want)
        print(
            f"PARITY {name} {tag:10s} max_norm {e['max_norm']:.3e} max_rel {e['max_rel']:.3e} "
            f"median_rel {e['median_rel']:.3e} fro {e['fro']:.3e}",
            flush=True,
        )
    empty = np.flatnonzero(active_sizes == 0)
    for e_idx in empty:
        print(
            f"PARITY {name} empty expert {e_idx}: |dw13| {float(jnp.abs(dw13[e_idx]).max()):.3e} "
            f"|dw2| {float(jnp.abs(dw2[e_idx]).max()):.3e}",
            flush=True,
        )


def parity_all():
    assert isinstance(_select_expert_mlp(fused_relu2, ungated=True), _CuteRelu2ExpertMlp), "selection did not pick QuACK"
    parity("small", E=4, H=256, I=384, O=256, physical_sizes=[300, 300, 300, 300], active_sizes=[257, 0, 300, 133])
    rng = np.random.default_rng(1)
    E, rows = 8, 65536
    active = rng.integers(int(0.75 * rows), rows, size=E).astype(np.int32)
    active[3] = 0
    parity("hero", E=E, H=3072, I=4608, O=3072, physical_sizes=[rows] * E, active_sizes=active.tolist())


def kernels():
    E, H, I, O, rows = 8, 3072, 4608, 3072, 65536
    C = E * rows
    ks = jax.random.split(jax.random.PRNGKey(6), 4)
    x = jax.random.normal(ks[0], (C, H), jnp.bfloat16) * 0.5
    wu = jax.random.normal(ks[1], (E, H, I), jnp.bfloat16) * H**-0.5
    wd = jax.random.normal(ks[2], (E, I, O), jnp.bfloat16) * I**-0.5
    dy = jax.random.normal(ks[3], (C, O), jnp.bfloat16)
    cu = jnp.arange(E + 1, dtype=jnp.int32) * rows
    up_fl, down_fl = 2 * C * H * I, 2 * C * I * O
    pre, post = jax.jit(
        lambda x, w: quack_act_grouped_gemm(x, w, cu, activation="relu_sq", **sonic_cute._QUACK_GATED_KW)
    )(x, wu)
    d_pre = pre

    def rep(tag, ms, fl):
        print(f"KERNEL {tag:70s} {ms:8.3f} ms {fl / (ms * 1e-3) / 1e12:8.1f} TF/s", flush=True)

    configs = [
        dict(tile_mn=(256, 256), cluster_mnk=(2, 1, 1), use_clc_persistence=True),
        dict(tile_mn=(256, 256), cluster_mnk=(2, 2, 1), use_clc_persistence=True),
        dict(tile_mn=(256, 128), cluster_mnk=(2, 1, 1), use_clc_persistence=True),
        dict(tile_mn=(256, 256), cluster_mnk=(2, 1, 1), use_clc_persistence=False),
    ]
    for kw in configs:
        tag = f"{kw['tile_mn']} {kw['cluster_mnk']} clc={kw['use_clc_persistence']}"
        for name, fn, args, fl in [
            (
                "act up fwd",
                lambda x, w, kw=kw: quack_act_grouped_gemm(x, w, cu, activation="relu_sq", **kw),
                (x, wu),
                up_fl,
            ),
            (
                "dact dpost",
                lambda d, w, p, kw=kw: quack_dact_grouped_gemm(d, w, p, cu, activation="relu_sq", **kw),
                (dy, wd, pre),
                down_fl,
            ),
            ("plain down fwd", lambda p, w, kw=kw: quack_grouped_gemm(p, w, cu, b_major="n", **kw), (post, wd), down_fl),
            ("plain dx", lambda d, w, kw=kw: quack_grouped_gemm(d, w, cu, b_major="k", **kw), (d_pre, wu), up_fl),
        ]:
            try:
                rep(f"{name} {tag}", bench(jax.jit(fn), *args), fl)
            except Exception as e:
                print(f"KERNEL {name} {tag} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
    rep(
        "wgrad dW_down (tuned)",
        bench(jax.jit(lambda a, b: quack_grouped_wgrad(a, b, cu, **sonic_cute._QUACK_WGRAD_KW)), post, dy),
        down_fl,
    )
    rep(
        "wgrad dW_up (tuned)",
        bench(jax.jit(lambda a, b: quack_grouped_wgrad(a, b, cu, **sonic_cute._QUACK_WGRAD_KW)), x, d_pre),
        up_fl,
    )


if __name__ == "__main__":
    print("devices", jax.devices()[:1], jax.devices()[0].device_kind, flush=True)
    which = sys.argv[1].split(",") if len(sys.argv) > 1 else ["parity", "kernels"]
    for w in which:
        {"parity": parity_all, "kernels": kernels}[w]()
