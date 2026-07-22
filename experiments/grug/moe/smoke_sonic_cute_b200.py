# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal on-hardware smoke for the sonic_cute (QuACK SM100) MoE backend on B200/GB200.

Builds a tiny d1280-shaped Grug MoE with ``moe_implementation="sonic_cute"`` and a
pure-JAX ``reference`` attention (so this exercises only the QuACK grouped GEMM, not the
SM90 fa4 kernels), then runs one fwd+bwd. Run on an allocated GB200 node with quack
installed. Prints ``SONIC_CUTE_SMOKE_OK`` on success.
"""

import traceback

import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import AxisType, Mesh

from experiments.grug.moe.model import GrugModelConfig, Transformer


def main() -> None:
    devs = jax.devices()
    print(f"jax {jax.__version__} devices={devs}", flush=True)
    n = len(devs)
    # sonic_cute is a local/no-EP backend: expert=1, shard the batch over data.
    mesh = Mesh(
        np.array(devs).reshape(1, n, 1, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    cfg = GrugModelConfig(
        vocab_size=512,
        hidden_dim=1280,
        num_layers=2,
        num_heads=10,
        num_kv_heads=2,
        head_dim=128,
        intermediate_dim=640,
        shared_expert_intermediate_dim=1280,
        num_experts=256,
        num_experts_per_token=4,
        max_seq_len=256,
        moe_implementation="sonic_cute",
        attention_implementation="reference",
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        tokens = jnp.ones((4, 128), dtype=jnp.int32)
        loss_weight = jnp.ones((4, 128), dtype=jnp.float32)

        # QuACK's GemmGatedSm100 requires 16-bit compute (bf16 postact); mirror training's
        # mixed-precision policy (params=float32, compute=bfloat16) by casting inside loss_fn.
        policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")

        def loss_fn(m):
            return policy.cast_to_compute(m).next_token_loss(tokens, loss_weight, reduction="mean")

        loss, grads = jax.value_and_grad(loss_fn)(model)
        jax.block_until_ready(grads)
        print(f"SONIC_CUTE_SMOKE_OK loss={float(loss):.4f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("SONIC_CUTE_SMOKE_FAIL", flush=True)
        traceback.print_exc()
        raise
