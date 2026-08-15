# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Layer-level A/B of Marin EP transports on one GB200 tray (EP4).

Times `marin_ep_moe_local` forward and forward+backward with
`transport="ragged"` (XLA `ragged_all_to_all` + local permutes) vs
`transport="mgpu"` (fused Mosaic-GPU puts, direct pool layout) at a
hero-like per-device shape scaled to 4 devices. Same waterfilling drop
rule and same `ragged_dot` GEMMs on both arms, so the delta is transport
+ permute elimination.

  uv run python experiments/marin_ep/bench/bench_backend_ep4.py [tokens_per_device]
"""

import functools
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.ep_marin import marin_ep_moe_local

HIDDEN = 3072  # hero routed latent width
INTERMEDIATE = 6272
NUM_EXPERTS = 192
TOPK = 4
CAPACITY_FACTOR = 1.33
POOL_GROUP = 3
ITERS = 20
ALPHA = 9.084  # hero-calibrated routing skew (MEP-H1)


def _bench(fn, args, label):
    fn(*args)  # compile + warm
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    ms = min(times) * 1e3
    print(f"  {label}: {ms:.3f} ms (median {np.median(times) * 1e3:.3f})")
    return ms


def main() -> None:
    tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
    devices = jax.device_count()
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 1, devices, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    batch_spec = P(("replica_dcn", "data", "expert"))
    weight_spec = P("expert", None, None)

    rng = np.random.default_rng(seed=0)
    probs = rng.dirichlet(np.full(NUM_EXPERTS, ALPHA))
    total = devices * tokens
    experts = rng.choice(NUM_EXPERTS, size=(total, TOPK), p=probs).astype(np.int32)
    x = rng.standard_normal((total, HIDDEN)).astype(np.float32)
    weights = rng.random((total, TOPK)).astype(np.float32)
    w13 = (0.02 * rng.standard_normal((NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE))).astype(np.float32)
    w2 = (0.02 * rng.standard_normal((NUM_EXPERTS, INTERMEDIATE, HIDDEN))).astype(np.float32)

    def put(a, spec, dtype):
        return jax.device_put(jnp.asarray(a, dtype=dtype), NamedSharding(mesh, spec))

    args = (
        put(x, batch_spec, jnp.bfloat16),
        put(experts, batch_spec, jnp.int32),
        put(weights, batch_spec, jnp.float32),
        put(w13, weight_spec, jnp.bfloat16),
        put(w2, weight_spec, jnp.bfloat16),
    )
    cot = put(rng.standard_normal((total, HIDDEN)), batch_spec, jnp.bfloat16)

    results = {}
    for transport in ("ragged", "mgpu"):
        shard_fn = shard_map(
            functools.partial(
                marin_ep_moe_local,
                activation_fn=jax.nn.silu,
                num_experts=NUM_EXPERTS,
                capacity_factor=CAPACITY_FACTOR,
                pool_group_size=POOL_GROUP,
                transport=transport,
            ),
            mesh=mesh,
            in_specs=(batch_spec, batch_spec, batch_spec, weight_spec, weight_spec),
            out_specs=(batch_spec, P()),
            check_vma=False,
        )

        def fwd(x_, e_, w_, w13_, w2_, _fn=shard_fn):
            return _fn(x_, e_, w_, w13_, w2_)

        def loss(x_, e_, w_, w13_, w2_, _fn=shard_fn):
            y, _ = _fn(x_, e_, w_, w13_, w2_)
            return jnp.sum((y * cot).astype(jnp.float32))

        with jax.set_mesh(mesh):
            print(f"transport={transport} tokens/device={tokens}")
            y, dropped = jax.jit(fwd)(*args)
            print(f"  dropped {int(dropped)} / {total * TOPK} ({100 * int(dropped) / (total * TOPK):.2f}%)")
            results[transport, "y"] = np.asarray(y, dtype=np.float32)
            results[transport, "fwd"] = _bench(jax.jit(fwd), args, "fwd")
            grad_fn = jax.jit(jax.grad(loss, argnums=(0, 3, 4)))
            results[transport, "fwdbwd"] = _bench(grad_fn, args, "fwd+bwd(grad x,w13,w2)")

    err = np.abs(results["ragged", "y"] - results["mgpu", "y"]).max()
    print(f"max |ragged - mgpu| forward output diff: {err:.4f}")
    for phase in ("fwd", "fwdbwd"):
        ragged_ms, mgpu_ms = results["ragged", phase], results["mgpu", phase]
        print(f"{phase}: ragged {ragged_ms:.3f} ms vs mgpu {mgpu_ms:.3f} ms -> {ragged_ms / mgpu_ms:.3f}x")


if __name__ == "__main__":
    main()
