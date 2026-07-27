#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare exact-clone MoE backward kernels on a single four-GPU GB200 node."""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.ep_ragged_all_to_all import _same_expert_cloned_fixed_a2a_core

_VARIANT_ENV = (
    "SCALE_A2A_CLONE_SONIC_SLOT_GATHER",
    "SCALE_A2A_CLONE_SONIC_WEIGHT_GRAD",
    "SCALE_QUACK_GROUPED_WGRAD",
)


def _set_variant(name: str) -> None:
    for key in _VARIANT_ENV:
        os.environ.pop(key, None)
    if name == "slot_gather":
        os.environ["SCALE_A2A_CLONE_SONIC_SLOT_GATHER"] = "1"
    elif name == "weight_reduce":
        os.environ["SCALE_A2A_CLONE_SONIC_WEIGHT_GRAD"] = "1"
    elif name == "grouped_wgrad":
        os.environ["SCALE_QUACK_GROUPED_WGRAD"] = "1"
    elif name == "combined":
        for key in _VARIANT_ENV:
            os.environ[key] = "1"
    elif name != "baseline":
        raise ValueError(f"unknown variant {name!r}")


def _gradient_metrics(actual, reference) -> dict[str, float | bool]:
    actual_leaves = jax.tree.leaves(actual)
    reference_leaves = jax.tree.leaves(reference)
    squared_error = sum(
        jnp.sum(jnp.square(a.astype(jnp.float32) - r.astype(jnp.float32)))
        for a, r in zip(actual_leaves, reference_leaves, strict=True)
    )
    squared_reference = sum(jnp.sum(jnp.square(r.astype(jnp.float32))) for r in reference_leaves)
    max_error = max(
        jnp.max(jnp.abs(a.astype(jnp.float32) - r.astype(jnp.float32)))
        for a, r in zip(actual_leaves, reference_leaves, strict=True)
    )
    finite = all(jnp.all(jnp.isfinite(a)) for a in actual_leaves)
    return {
        "relative_l2": float(jnp.sqrt(squared_error / jnp.maximum(squared_reference, 1e-30))),
        "max_abs": float(max_error),
        "finite": bool(finite),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-device", type=int, default=8192)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["slot_gather", "weight_reduce", "grouped_wgrad", "combined"],
    )
    args = parser.parse_args()

    devices = jax.devices()
    if len(devices) != 4:
        raise ValueError(f"expected four GPUs, got {len(devices)}")

    os.environ.update(
        {
            "SCALE_A2A_CLONE_POOLED": "1",
            "SCALE_A2A_CLONE_SPARSE_WEIGHTS": "1",
            "SCALE_A2A_CLONE_MAX_RECEIVER_EXPERTS": "16",
            "SCALE_A2A_CLONE_TOKEN_PADDING_EXPERTS": "0",
            "SCALE_A2A_SONIC_DISPATCH": "1",
            "SCALE_A2A_CLONE_SONIC_CUTE": "1",
            "SCALE_A2A_SONIC_COMBINE": "1",
            "SCALE_A2A_NO_BARRIER": "1",
        }
    )

    expert_shards = len(devices)
    num_experts = 16
    topk = 8
    hidden_dim = 5120
    intermediate_dim = 1280
    tokens = args.tokens_per_device * expert_shards
    mesh = Mesh(
        np.asarray(devices),
        axis_names=("expert",),
        axis_types=(AxisType.Explicit,),
    )
    batch_sharding = NamedSharding(mesh, P("expert", None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))

    token_ids = jnp.arange(tokens, dtype=jnp.int32)
    selected_experts = jnp.stack(
        [
            jnp.zeros_like(token_ids),
            *[1 + (token_ids * 5 + offset * 2) % (num_experts - 1) for offset in range(topk - 1)],
        ],
        axis=1,
    )
    logits = jax.random.normal(jax.random.key(1), (tokens, topk), dtype=jnp.float32)
    combine_weights = jax.nn.softmax(logits, axis=-1)
    x = jax.random.normal(jax.random.key(2), (tokens, hidden_dim), dtype=jnp.bfloat16) * jnp.bfloat16(0.02)
    output_cotangent = jax.random.normal(jax.random.key(3), (tokens, hidden_dim), dtype=jnp.bfloat16) * jnp.bfloat16(
        0.01
    )
    w13 = jax.random.normal(
        jax.random.key(4),
        (num_experts, hidden_dim, 2 * intermediate_dim),
        dtype=jnp.bfloat16,
    ) * jnp.bfloat16(0.007)
    w2 = jax.random.normal(
        jax.random.key(5),
        (num_experts, intermediate_dim, hidden_dim),
        dtype=jnp.bfloat16,
    ) * jnp.bfloat16(0.007)

    with jax.set_mesh(mesh):
        inputs = (
            jax.device_put(x, batch_sharding),
            jax.device_put(selected_experts, batch_sharding),
            jax.device_put(combine_weights, batch_sharding),
            jax.device_put(w13, expert_sharding),
            jax.device_put(w2, expert_sharding),
            jax.device_put(output_cotangent, batch_sharding),
        )

        def run(name: str):
            _set_variant(name)

            def local_loss(x_local, selected_local, weights_local, w13_local, w2_local, cotangent_local):
                output, dropped = _same_expert_cloned_fixed_a2a_core(
                    x_local,
                    selected_local,
                    weights_local,
                    w13_local,
                    w2_local,
                    activation_fn=jax.nn.silu,
                    num_experts=num_experts,
                    capacity_factor=1.0,
                )
                local_value = jnp.sum(output.astype(jnp.float32) * cotangent_local.astype(jnp.float32))
                return jax.lax.psum(local_value, "expert"), dropped

            sharded_loss = jax.shard_map(
                local_loss,
                mesh=mesh,
                in_specs=(P("expert"),) * 6,
                out_specs=(P(), P()),
                check_vma=False,
            )
            return jax.jit(jax.value_and_grad(sharded_loss, argnums=(0, 2, 3, 4), has_aux=True))(*inputs)

        (baseline_value, baseline_dropped), baseline_grad = run("baseline")
        baseline_value.block_until_ready()
        print(
            {
                "variant": "baseline",
                "value": float(baseline_value),
                "dropped": int(baseline_dropped),
                "finite": bool(all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(baseline_grad))),
            },
            flush=True,
        )

        for variant in args.variants:
            (value, dropped), gradient = run(variant)
            value.block_until_ready()
            gradient_metrics = {
                name: _gradient_metrics(actual, reference)
                for name, actual, reference in zip(
                    ("x", "combine_weights", "w13", "w2"),
                    gradient,
                    baseline_grad,
                    strict=True,
                )
            }
            print(
                {
                    "variant": variant,
                    "value": float(value),
                    "value_delta": float(value - baseline_value),
                    "dropped": int(dropped),
                    "gradient": gradient_metrics,
                },
                flush=True,
            )


if __name__ == "__main__":
    main()
