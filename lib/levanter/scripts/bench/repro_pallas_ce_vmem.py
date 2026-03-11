# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal reproducer for TPU Pallas fused cross-entropy scoped-vmem failures.

This reproduces the kernel shape from the Grug MoE perf experiments without the
rest of the model stack. It is intended to be run on a TPU worker host.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
from jax import random

from levanter.kernels.pallas.fused_cross_entropy_loss import (
    BlockSizes,
    fused_cross_entropy_loss_and_logsumexp_penalty,
)
from levanter.kernels.pallas.fused_cross_entropy_loss.tuned_block_sizes import infer_block_sizes_with_tuned_match


def _dtype(name: str) -> jnp.dtype:
    return {
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
    }[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=40_960)
    parser.add_argument("--hidden", type=int, default=2_048)
    parser.add_argument("--vocab", type=int, default=128_256)
    parser.add_argument("--x-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--w-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--compute-dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--implementation", default="pallas_tpu")
    parser.add_argument("--v-block-divisor", type=int, default=1)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run_repro(
    *,
    batch: int,
    hidden: int,
    vocab: int,
    x_dtype_name: str,
    w_dtype_name: str,
    compute_dtype_name: str,
    implementation: str,
    v_block_divisor: int,
    backward: bool,
    seed: int,
) -> float:
    if jax.default_backend() != "tpu":
        raise RuntimeError("This reproducer is intended to run on TPU.")

    x_dtype = _dtype(x_dtype_name)
    w_dtype = _dtype(w_dtype_name)
    compute_dtype = _dtype(compute_dtype_name)
    tuned, has_tuned_match = infer_block_sizes_with_tuned_match(
        batch,
        hidden,
        vocab,
        dtype=compute_dtype,
    )
    v_block_size = max(128, tuned.v_block_size // v_block_divisor)
    v_block_size = max(128, (v_block_size // 128) * 128)
    block_sizes = BlockSizes(
        b_block_size=tuned.b_block_size,
        h_block_size=tuned.h_block_size,
        v_block_size=v_block_size,
    )

    print(
        "Running fused CE repro:",
        {
            "backend": jax.default_backend(),
            "shape": (batch, hidden, vocab),
            "x_dtype": str(x_dtype),
            "w_dtype": str(w_dtype),
            "compute_dtype": str(compute_dtype),
            "implementation": implementation,
            "tuned_block_sizes": tuned,
            "has_tuned_match": has_tuned_match,
            "block_sizes": block_sizes,
        },
    )

    key = random.PRNGKey(seed)
    kx, kw, kl = random.split(key, 3)
    x = random.normal(kx, (batch, hidden), dtype=x_dtype)
    w = random.normal(kw, (hidden, vocab), dtype=w_dtype)
    labels = random.randint(kl, (batch,), 0, vocab, dtype=jnp.int32)

    def loss_fn(x: jax.Array, w: jax.Array) -> jax.Array:
        return fused_cross_entropy_loss_and_logsumexp_penalty(
            x,
            labels,
            w,
            reduction="mean",
            dtype=compute_dtype,
            implementation=implementation,
            block_sizes=block_sizes,
        )

    if backward:

        @jax.jit
        def run(x: jax.Array, w: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            return jax.value_and_grad(loss_fn, argnums=(0, 1))(x, w)

        loss, (grad_x, grad_w) = run(x, w)
        loss, grad_x, grad_w = jax.block_until_ready((loss, grad_x, grad_w))
        value = float(loss)
        print("loss", value)
        print("grad_x_norm", float(jnp.linalg.norm(grad_x.astype(jnp.float32))))
        print("grad_w_norm", float(jnp.linalg.norm(grad_w.astype(jnp.float32))))
    else:

        @jax.jit
        def run(x: jax.Array, w: jax.Array) -> jax.Array:
            return loss_fn(x, w)

        loss = run(x, w)
        value = float(loss.block_until_ready())
        print("loss", value)
    return value


def main() -> None:
    args = parse_args()
    run_repro(
        batch=args.batch,
        hidden=args.hidden,
        vocab=args.vocab,
        x_dtype_name=args.x_dtype,
        w_dtype_name=args.w_dtype,
        compute_dtype_name=args.compute_dtype,
        implementation=args.implementation,
        v_block_divisor=args.v_block_divisor,
        backward=args.backward,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
