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
from haliax.nn import ArrayStacked
from iris.runtime.jax_init import initialize_jax
from jax import random
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.grug_moe import MOE_REMAT_SAVE_NAMES
from levanter.kernels.mixture_of_kittens import (
    MokLikeBuildConfig,
    MokLikeConfig,
    MokLikeWorkspaceTransport,
    initialize_mok_like_runtime,
)

from experiments.grug.moe_hero_ep.launch_mok_like import MOK_LIKE_BUILD_ROOT, MOK_LIKE_SOURCE_ROOT
from experiments.grug.moe_hero_ep.model import DenseMLP, GrugModelConfig, MoEMLP

WORLD_SIZE = 4


@click.command()
@click.option("--hidden-dim", type=int, default=512, show_default=True)
@click.option("--intermediate-dim", type=int, default=512, show_default=True)
@click.option("--num-tokens", type=int, default=512, show_default=True, help="tokens per rank")
@click.option(
    "--scan-layers", type=int, default=1, show_default=True, help="run the block under lax.scan this many times"
)
@click.option("--backward", is_flag=True, help="also take a gradient through the block")
@click.option(
    "--remat",
    is_flag=True,
    help="scan over stacked per-layer parameters with eqx.filter_checkpoint applied per iteration, "
    "matching Transformer._scan_layers; implies a backward pass, which is what makes remat replay",
)
def main(hidden_dim: int, intermediate_dim: int, num_tokens: int, scan_layers: int, backward: bool, remat: bool) -> None:
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
        vocab_size=1024,
        remat_mode="save_moe",
        mok_like=mok_like,
    )

    with jax.sharding.set_mesh(mesh):
        # The trainer holds parameters in float32 and casts to bfloat16 through its mixed-precision
        # policy before the block runs; the kernel requires bfloat16, so apply the same cast here.
        def to_bf16(tree):
            return jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf, tree)

        block = to_bf16(MoEMLP.init(cfg, key=random.PRNGKey(0)))
        shared = to_bf16(DenseMLP.init(hidden_dim, intermediate_dim, cfg.initializer_std, key=random.PRNGKey(1)))
        # The block reshards its output onto the batch spec, so a scan carry only type-checks if
        # the input already carries the same sharding.
        batch_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None, None))
        tokens = jax.device_put(
            jnp.asarray(
                np.random.default_rng(7).normal(size=(WORLD_SIZE, num_tokens, hidden_dim)),
                dtype=jnp.bfloat16,
            ),
            batch_sharding,
        )

        with initialize_mok_like_runtime(
            build_config=MokLikeBuildConfig(
                source_root=MOK_LIKE_SOURCE_ROOT,
                cache_root=MOK_LIKE_BUILD_ROOT,
                cuda_arch="sm_100a",
                clone_if_missing=True,
                num_devices=WORLD_SIZE,
            ),
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

            if remat:
                # `Transformer._scan_layers` applies `eqx.filter_checkpoint` per scan iteration with
                # the stacked per-layer parameters riding in as `xs`. A rematerialized forward is
                # replayed on the backward pass, so the FFI runs a second time and takes a second
                # set of workspace reservations; whether the ranks agree on their order is exactly
                # what none of the earlier probes exercised. This is the last structural difference
                # between them and the training path, so it needs the backward pass to mean anything.
                stacked = to_bf16(
                    ArrayStacked.init(scan_layers, MoEMLP)(
                        cfg, key=jnp.stack([random.PRNGKey(seed) for seed in range(scan_layers)])
                    )
                )
                remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)

                def remat_scanned(layers: MoEMLP, shared_expert: DenseMLP, x: jax.Array) -> jax.Array:
                    def body(carry: jax.Array, layer: MoEMLP) -> tuple[jax.Array, None]:
                        output, _ = eqx.filter_checkpoint(layer, policy=remat_policy)(
                            carry, shared_expert=shared_expert, mok_like_runtime=runtime
                        )
                        return output.astype(carry.dtype), None

                    final, _ = jax.lax.scan(body, x, xs=layers)
                    return final

                def remat_loss(layers: MoEMLP, shared_expert: DenseMLP, x: jax.Array) -> jax.Array:
                    return jnp.sum(remat_scanned(layers, shared_expert, x).astype(jnp.float32))

                activation_grad = jax.jit(jax.grad(remat_loss, argnums=2))(stacked.stacked, shared, tokens)
                activation_grad.block_until_ready()
                activation_finite = bool(jnp.all(jnp.isfinite(activation_grad.astype(jnp.float32))))
                print(f"REMAT activation-grad layers={scan_layers} finite={activation_finite}", flush=True)
                if not activation_finite:
                    raise RuntimeError("remat scan produced a non-finite activation gradient")

                # The trainer differentiates the parameters, which is what puts the stacked `xs`
                # leaves on the backward path rather than only the carry. Split the arrays off the
                # module first: the static half carries the config, whose dict field is unhashable,
                # so handing the whole module to filter_grad fails before it ever reaches the kernel.
                diff_layers, static_layers = eqx.partition(stacked.stacked, eqx.is_inexact_array)

                def parameter_loss(layers: MoEMLP, shared_expert: DenseMLP, x: jax.Array) -> jax.Array:
                    return remat_loss(eqx.combine(layers, static_layers), shared_expert, x)

                parameter_grads = jax.jit(jax.grad(parameter_loss))(diff_layers, shared, tokens)
                parameter_leaves = [leaf for leaf in jax.tree.leaves(parameter_grads) if eqx.is_inexact_array(leaf)]
                jax.block_until_ready(parameter_leaves)
                parameter_finite = all(
                    bool(jnp.all(jnp.isfinite(leaf.astype(jnp.float32)))) for leaf in parameter_leaves
                )
                print(
                    f"REMAT parameter-grad leaves={len(parameter_leaves)} finite={parameter_finite}",
                    flush=True,
                )
                if not parameter_finite:
                    raise RuntimeError("remat scan produced a non-finite parameter gradient")
                print("REMAT PASS", flush=True)

            if scan_layers > 1:
                # The transformer runs its layers under lax.scan, so the FFI is compiled once and
                # executed N times from inside a traced body -- not N separate calls. That is the
                # last structural difference between these probes and the training path.
                def scanned(module: MoEMLP, shared_expert: DenseMLP, x: jax.Array) -> jax.Array:
                    def body(carry: jax.Array, _: None) -> tuple[jax.Array, None]:
                        return forward(module, shared_expert, carry).astype(carry.dtype), None

                    final, _ = jax.lax.scan(body, x, None, length=scan_layers)
                    return final

                scanned_output = jax.jit(scanned)(block, shared, tokens)
                scanned_output.block_until_ready()
                scanned_finite = bool(jnp.all(jnp.isfinite(scanned_output.astype(jnp.float32))))
                print(f"SCAN layers={scan_layers} finite={scanned_finite}", flush=True)
                if not scanned_finite:
                    raise RuntimeError("scanned block produced a non-finite output")
                print("SCAN PASS", flush=True)

            output = jax.jit(forward)(block, shared, tokens)
            output.block_until_ready()
            finite = bool(jnp.all(jnp.isfinite(output.astype(jnp.float32))))
            print(f"BLOCK forward shape={output.shape} finite={finite}", flush=True)
            if not finite:
                raise RuntimeError("block forward produced a non-finite output")
            print("BLOCK FORWARD PASS", flush=True)

            if backward:

                # Differentiate with respect to the activations. The module carries its config in
                # the tree, which filter_grad wants to treat as a static hashable, so grad through
                # the input instead -- it still drives the kernel's backward handler.
                def loss(module: MoEMLP, shared_expert: DenseMLP, x: jax.Array) -> jax.Array:
                    return jnp.sum(forward(module, shared_expert, x).astype(jnp.float32))

                gradient = jax.jit(jax.grad(loss, argnums=2))(block, shared, tokens)
                gradient.block_until_ready()
                all_finite = bool(jnp.all(jnp.isfinite(gradient.astype(jnp.float32))))
                print(f"BLOCK backward shape={gradient.shape} finite={all_finite}", flush=True)
                if not all_finite:
                    raise RuntimeError("block backward produced a non-finite gradient")
                print("BLOCK BACKWARD PASS", flush=True)


if __name__ == "__main__":
    main()
