"""Single-GPU check of XLA's async-stream assignment for an offloaded layer carry.

A rematted layer scan whose carry is offloaded to pinned host, with several stacked weights, gives
the forward body one carry dynamic-update-slice to host plus one async weight dynamic-slice per
weight, and the backward body the mirror image. Run under
TF_CPP_VMODULE=execution_stream_assignment=3 and grep "Start new compute execution scope" to see
which stream each async start gets. With XLA_GPU_HOST_TRANSFER_STREAMS=1 on a patched PJRT plugin the
carry transfers land on dedicated streams past the round-robin pool. Also checks that the gradient is
bitwise identical across the two modes (pass --reference <npy> from the first run).

    uv run python stream_smoke.py <out.npy> [--reference <npy>]
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.ad_checkpoint import checkpoint_name

LAYERS = 8
TOKENS = 16384
WIDTH = 4096
NUM_WEIGHTS = 6
CARRY = "carry"

POLICY = jax.checkpoint_policies.save_and_offload_only_these_names(
    names_which_can_be_saved=[],
    names_which_can_be_offloaded=[CARRY],
    offload_src="device",
    offload_dst="pinned_host",
)


def layer(x, weights):
    x = checkpoint_name(x, CARRY)
    h = x
    for w in weights:
        h = jnp.tanh(h @ w)
    return x + h


def loss(weights, x):
    def body(carry, layer_weights):
        return jax.checkpoint(layer, policy=POLICY)(carry, layer_weights), None

    y, _ = jax.lax.scan(body, x, weights)
    return jnp.mean(jnp.square(y.astype(jnp.float32)))


def main(out_path: str, reference: str | None = None) -> None:
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, NUM_WEIGHTS + 1)
    weights = tuple(
        (jax.random.normal(k, (LAYERS, WIDTH, WIDTH), jnp.float32) / WIDTH**0.5).astype(jnp.bfloat16)
        for k in keys[:NUM_WEIGHTS]
    )
    x = jax.random.normal(keys[-1], (TOKENS, WIDTH), jnp.float32).astype(jnp.bfloat16)
    grad_fn = jax.jit(jax.grad(loss))
    grads = jax.block_until_ready(grad_fn(weights, x))
    times = []
    for _ in range(5):
        start = time.perf_counter()
        grads = jax.block_until_ready(grad_fn(weights, x))
        times.append(time.perf_counter() - start)
    print(f"step ms: median {1e3 * sorted(times)[len(times) // 2]:.2f} all {[round(1e3 * t, 2) for t in times]}")
    flat = np.concatenate([np.asarray(g, dtype=np.float32).ravel() for g in grads])
    np.save(out_path, flat)
    if reference is not None:
        ref = np.load(reference)
        print("bitwise equal to reference:", bool(np.array_equal(ref, flat)), "max abs diff", float(np.max(np.abs(ref - flat))))


if __name__ == "__main__":
    args = sys.argv[1:]
    ref = None
    if "--reference" in args:
        i = args.index("--reference")
        ref = args[i + 1]
        args = args[:i] + args[i + 2 :]
    main(args[0], ref)
