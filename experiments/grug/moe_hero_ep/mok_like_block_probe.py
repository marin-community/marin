# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the hero model's MoE block against the fabric runtime, one layer, no trainer.

The kernel-level gate passes cross-process on every dimension the training path exercises --
shape, repeated and chained invocations, backward, offloaded context, single-slot reuse, and
adversarial routing -- while the training path faults with an illegal address. That places the
fault in model-level integration rather than the transport, and this probe is the smallest thing
that includes the model: a real `MoEMLP` with its own router, dispatching through `mok_like`.

It compiles one block instead of a training step, which is the difference between a minute and an
hour per attempt.
"""

import click
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from iris.runtime.jax_init import initialize_jax
from jax import random
from jax.sharding import AxisType, Mesh
from levanter.kernels.mixture_of_kittens import (
    MokLikeBuildConfig,
    MokLikeConfig,
    MokLikeWorkspaceTransport,
    initialize_mok_like_runtime,
)

from experiments.grug.moe_hero_ep.model import DenseMLP, GrugModelConfig, MoEMLP

WORLD_SIZE = 4


@click.command()
@click.option("--hidden-dim", type=int, default=512, show_default=True)
@click.option("--intermediate-dim", type=int, default=512, show_default=True)
@click.option("--num-tokens", type=int, default=512, show_default=True, help="tokens per rank")
@click.option("--backward", is_flag=True, help="also take a gradient through the block")
def main(hidden_dim: int, intermediate_dim: int, num_tokens: int, backward: bool) -> None:
    initialize_jax()
    devices = jax.devices()
    if jax.process_count() != WORLD_SIZE or jax.local_device_count() != 1:
        raise RuntimeError(
            f"needs {WORLD_SIZE} processes with one GPU each, got {jax.process_count()} and "
            f"{jax.local_device_count()}"
        )

    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, WORLD_SIZE, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )

    mok_like = MokLikeConfig(workspace_transport=MokLikeWorkspaceTransport.FABRIC_SYMMETRIC)
    cfg = GrugModelConfig(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        shared_expert_intermediate_dim=intermediate_dim,
        num_experts=WORLD_SIZE * 2,
        num_experts_per_token=4,
        num_shared_experts=1,
        num_layers=1,
        mok_like=mok_like,
    )

    with jax.sharding.use_mesh(mesh):
        block = MoEMLP.init(cfg, key=random.PRNGKey(0))
        shared = DenseMLP.init(hidden_dim, intermediate_dim, cfg.initializer_std, key=random.PRNGKey(1))
        tokens = jnp.asarray(
            np.random.default_rng(7).normal(size=(1, WORLD_SIZE * num_tokens, hidden_dim)),
            dtype=jnp.bfloat16,
        )

        with initialize_mok_like_runtime(
            build_config=MokLikeBuildConfig(num_devices=WORLD_SIZE),
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            top_k=cfg.num_experts_per_token,
            workspace_slots=mok_like.workspace_slots,
            mesh=mesh,
            workspace_transport=MokLikeWorkspaceTransport.FABRIC_SYMMETRIC,
        ) as runtime:

            def forward(module: MoEMLP, shared_expert: DenseMLP, x: jax.Array) -> jax.Array:
                output, _ = module(x, shared_expert=shared_expert, mok_like_runtime=runtime)
                return output

            output = jax.jit(forward)(block, shared, tokens)
            output.block_until_ready()
            finite = bool(jnp.all(jnp.isfinite(output.astype(jnp.float32))))
            print(f"BLOCK forward shape={output.shape} finite={finite}", flush=True)
            if not finite:
                raise RuntimeError("block forward produced a non-finite output")
            print("BLOCK FORWARD PASS", flush=True)

            if backward:

                def loss(module: MoEMLP, shared_expert: DenseMLP, x: jax.Array) -> jax.Array:
                    return jnp.sum(forward(module, shared_expert, x).astype(jnp.float32))

                gradients = jax.jit(eqx.filter_grad(loss))(block, shared, tokens)
                jax.block_until_ready(gradients)
                leaves = [leaf for leaf in jax.tree.leaves(gradients) if eqx.is_inexact_array(leaf)]
                all_finite = all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
                print(f"BLOCK backward leaves={len(leaves)} finite={all_finite}", flush=True)
                if not all_finite:
                    raise RuntimeError("block backward produced a non-finite gradient")
                print("BLOCK BACKWARD PASS", flush=True)


if __name__ == "__main__":
    main()
