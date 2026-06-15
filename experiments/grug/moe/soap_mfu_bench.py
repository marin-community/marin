# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Microbench for the KLSOAPH optimizer step — the engine for the MFU hill-climb.

Times ONLY ``scale_by_klsoaph().update()`` on the real d512 expert-leaf shapes under a
realistic (data, expert) mesh, so MFU-affecting variants (sharding, precond_freq, dtype)
can be compared in minutes instead of waiting on a ~45-min full-model compile. The
optimizer step is the SOAP overhead that sits on top of fwd/bwd; shrinking it is the MFU
lever. Reports compile time + mean per-step time over the expert leaves.

Run on a TPU (e.g. v5p-8):  python -m experiments.grug.moe.soap_mfu_bench
Env knobs: KLSOAPH_PRECOND_FREQ, KLSOAPH_BETA1, MFU_LAYERS, MFU_STEPS.
"""

import os
import time

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe.klsoaph import scale_by_klsoaph

# d512 May-recipe expert leaves (from MoEExpertMlpPspecs): w_gate_up [E, hidden, 2*inter],
# w_down [E, inter, hidden]. E=256, hidden=512, inter=256.
_E, _HID, _INTER = 256, 512, 256
_LAYERS = int(os.environ.get("MFU_LAYERS", "6"))
_STEPS = int(os.environ.get("MFU_STEPS", "30"))
_FREQ = int(os.environ.get("KLSOAPH_PRECOND_FREQ", "1"))
_BETA1 = float(os.environ.get("KLSOAPH_BETA1", "0.95"))
_BLOCK = int(os.environ.get("KLSOAPH_BLOCK_SIZE", "0"))


def _build_mesh():
    n = len(jax.devices())
    # Mirror v5p-8: split into (data, expert). expert takes 2 if possible.
    expert = 2 if n % 2 == 0 else 1
    data = n // expert
    return jax.make_mesh((data, expert), ("data", "expert"), axis_types=(jax.sharding.AxisType.Explicit,) * 2)


def _params():
    p = {}
    for i in range(_LAYERS):
        p[f"l{i}.w_gate_up"] = jnp.zeros((_E, _HID, 2 * _INTER), jnp.float32)
        p[f"l{i}.w_down"] = jnp.zeros((_E, _INTER, _HID), jnp.float32)
    return p


def _state_layout_fn(layout):
    """Reshard one optimizer-state leaf to the target storage layout. ``replicate`` mimics a trainer
    that stores opt_state un-sharded (each step pays a replicated->mat_p->replicated round trip);
    ``persist`` stores it expert-sharded (leading axis over all mesh axes) so the per-step reshards
    in _klsoaph_step_sharded are no-ops. The delta isolates the per-step state-reshard cost."""

    def fn(x):
        if not hasattr(x, "ndim") or x.ndim == 0:
            return jax.sharding.reshard(x, P())
        if layout == "replicate":
            return jax.sharding.reshard(x, P(*((None,) * x.ndim)))
        return jax.sharding.reshard(x, P(("data", "expert"), *((None,) * (x.ndim - 1))))  # persist: shard E

    return fn


def _time_layout(layout, opt, params, mesh, key):
    """Time opt_step with opt_state STORED in `layout` between steps (re-jit so the out-sharding differs)."""
    shard_state = _state_layout_fn(layout)

    def shard(x):
        spec = P("expert", "data", None) if x.ndim == 3 else P(*((None,) * x.ndim))
        return jax.device_put(x, NamedSharding(mesh, spec))

    def mkgrad(k):
        return {kk: shard(jax.random.normal(jax.random.fold_in(k, hash(kk) % 997), v.shape)) for kk, v in params.items()}

    def step(g, s):
        u, ns = opt.update(g, s)
        return u, jax.tree.map(shard_state, ns)  # force the trainer's storage layout each step

    jstep = jax.jit(step)
    state = jax.tree.map(shard_state, opt.init(params))

    g0 = mkgrad(key)
    t0 = time.perf_counter()
    u, state = jstep(g0, state)
    jax.block_until_ready(u)
    t_compile = time.perf_counter() - t0

    for i in range(10):
        u, state = jstep(mkgrad(jax.random.fold_in(key, i + 1)), state)
    jax.block_until_ready(u)
    t0 = time.perf_counter()
    for i in range(_STEPS):
        u, state = jstep(mkgrad(jax.random.fold_in(key, 100 + i)), state)
    jax.block_until_ready(u)
    return t_compile, (time.perf_counter() - t0) / _STEPS


def main():
    mesh = _build_mesh()
    print(f"dev={len(jax.devices())} mesh={dict(mesh.shape)} L={_LAYERS} freq={_FREQ} b1={_BETA1} block={_BLOCK}")
    params = _params()
    key = jax.random.PRNGKey(0)
    opt = scale_by_klsoaph(
        beta1=_BETA1, beta2=0.9, shampoo_beta=0.9, eps=1e-8, precond_freq=_FREQ, init_factor=0.1, block_size=_BLOCK
    )

    with jax.set_mesh(mesh):
        results = {layout: _time_layout(layout, opt, params, mesh, key) for layout in ("replicate", "persist")}

    rc, rs = results["replicate"]
    pc, ps = results["persist"]
    print(f"RESULT replicate-state: compile={rc:.1f}s opt_step={rs * 1e3:.1f}ms")
    print(f"RESULT persist-state:   compile={pc:.1f}s opt_step={ps * 1e3:.1f}ms")
    print(f"RESULT speedup (replicate/persist) = {rs / ps:.2f}x  (reshard cost removed = {(rs - ps) * 1e3:.1f}ms/step)")


if __name__ == "__main__":
    main()
