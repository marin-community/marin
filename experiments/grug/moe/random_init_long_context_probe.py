# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Random-init 67B long-context forward probe on a 16-device TPU slice."""

import argparse
import dataclasses
import functools
import json
import time
from typing import cast

import jax
import jax.numpy as jnp
import jmp
from haliax.partitioning import set_mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.evaluate import _canonical_67b_model
from experiments.grug.moe.model import Transformer

_BATCH_AXES = ("replica_dcn", "data", "expert")
_MP_POLICY = "params=float32,compute=bfloat16,output=bfloat16"


def run_probe(*, seq_len: int, batch_size: int, qk_mult: float) -> None:
    """Initialize the canonical 67B model and run one synthetic loss pass."""

    if seq_len <= 0 or batch_size <= 0:
        raise ValueError("seq_len and batch_size must be positive")
    model_config = dataclasses.replace(
        _canonical_67b_model(),
        max_seq_len=seq_len,
        qk_mult=qk_mult,
    )
    mesh = compact_grug_mesh(
        expert_axis_size=1,
        replica_axis_size=1,
        model_axis_size=1,
        context_axis_size=1,
    )
    if mesh.shape["data"] != batch_size:
        raise ValueError(f"batch_size={batch_size} must equal the resolved data axis {mesh.shape['data']}")

    policy = jmp.get_policy(_MP_POLICY)
    started = time.perf_counter()
    with set_mesh(mesh):
        model = jax.jit(lambda key: policy.cast_to_param(Transformer.init(model_config, key=key)))(jax.random.PRNGKey(0))
        init_seconds = time.perf_counter() - started

        batch_sharding = NamedSharding(mesh, P(_BATCH_AXES, None))

        @functools.partial(jax.jit, out_shardings=(batch_sharding, batch_sharding))
        def synthetic_inputs() -> tuple[jax.Array, jax.Array]:
            tokens = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
            loss_weight = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
            return tokens, loss_weight

        tokens, loss_weight = synthetic_inputs()

        @jax.jit
        def forward(params: Transformer, input_ids: jax.Array, weights: jax.Array) -> jax.Array:
            return cast(
                jax.Array,
                params.next_token_loss(
                    input_ids,
                    weights,
                    mask=AttentionMask.causal(),
                    reduction="mean",
                    logsumexp_weight=None,
                ),
            )

        forward_started = time.perf_counter()
        loss = forward(policy.cast_to_compute(model), tokens, loss_weight)
        loss = jax.block_until_ready(loss)
        forward_seconds = time.perf_counter() - forward_started

    if jax.process_index() == 0:
        print(
            json.dumps(
                {
                    "batch_size": batch_size,
                    "device_count": jax.device_count(),
                    "forward_seconds": forward_seconds,
                    "init_seconds": init_seconds,
                    "loss": float(loss),
                    "mesh": dict(mesh.shape),
                    "qk_mult": qk_mult,
                    "seq_len": seq_len,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=32_768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--qk-mult", type=float, default=1.75)
    args = parser.parse_args()
    run_probe(seq_len=args.seq_len, batch_size=args.batch_size, qk_mult=args.qk_mult)
