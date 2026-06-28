import functools
import itertools
import math
from dataclasses import dataclass
from typing import Callable

import jax
import numpy as np
from jax import lax, random
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import profiler
from jax.experimental.pallas import mosaic_gpu as mgpu
from jaxtyping import Array, Float, Int


@dataclass(frozen=True)
class MoeMgpuConfig:
    block_m: int = 128
    block_n: int = 128
    block_k: int = 64
    max_concurrent_steps: int = 2
    grid_block_n: int = 8
    capacity_factor: float = 1.2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _MoeMgpuMetadata:
    token_idx: Int[Array, "localTK"]  # Original token index for each routed row. Linearized inside TK
    global_expert_counts: Int[Array, "EP E"]  # Global expert counts across all ranks


@dataclass(frozen=True)
class GroupInfo:
    """Information regarding the group being processed in a block."""

    group_id: jax.Array
    block: jax.Array
    block_start: jax.Array
    actual_start: jax.Array
    actual_end: jax.Array
    start_within_block: jax.Array
    actual_size: jax.Array

    @classmethod
    def create(cls, group_lengths, tile, tid):
        """Get the group info for the current block."""

        tile = jnp.int32(tile)
        group_boundaries = [group_lengths[i] for i in range(len(group_lengths))]

        # We usually only have very few groups, so we unroll the loop processing
        # them. Normally we'd break out of the loop early, once we'd have found our
        # boundary, but we can't do that when unrolling, so we rely on many selects
        # to mask out the epilogue of the loop.
        group_end = group_start = block = group = end = jnp.array(0, dtype=jnp.int32)

        for i, b in enumerate(group_boundaries):
            # Start/end are inclusive
            start = end
            end = start + b
            final = end - 1
            start_block = lax.div(start, tile)
            final_block = lax.div(final, tile)
            block_end = final_block + 1
            tid_begin = start_block + i
            tid_end = block_end + i
            # How many blocks after is our block?
            this_is_group = (tid_begin <= tid) & (tid < tid_end)
            block = lax.select(this_is_group, tid - tid_begin + start_block, block)
            group = lax.select(this_is_group, jnp.int32(i), group)
            group_start = lax.select(this_is_group, start, group_start)
            group_end = lax.select(this_is_group, end, group_end)

        block_start = block * tile
        actual_start = jnp.maximum(group_start, block_start)
        actual_end = jnp.minimum(group_end, block_start + tile)
        start_within_block = actual_start - block_start
        actual_size = actual_end - actual_start
        return cls(
            group_id=group,
            block=block,
            block_start=block_start,
            actual_start=actual_start,
            actual_end=actual_end,
            start_within_block=start_within_block,
            actual_size=actual_size,
        )


def moe_mlp_mgpu(
    x: Float[Array, "Tlocal D"],
    selected_experts: Int[Array, "Tlocal K"],
    combine_weights: Float[Array, "Tlocal K"],
    moe_w13: Float[Array, "Elocal D I2"],
    moe_w2: Float[Array, "Elocal I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    expert_axis: str = "expert",
    config: MoeMgpuConfig = MoeMgpuConfig(),
):
    """
    Computes the MoE MLP operation using Mosaic GPU primitives that is:
        dispatch(x, selected_experts) -> @moe_w13 -> activation_fn -> @moe_w2 -> combine(combine_weights) -> output

    Basic approach:
        1) compute dispatch indices locally upfront and broadcast them
        2) fuse dispatch/w13/activation into a single kernel by sending blocks over nvlink
        3) fuse w2/combine into a single kernel by sending blocks and processing them
    """

    T, D = x.shape
    localE, D, I2 = moe_w13.shape
    I = I2 // 2
    K = selected_experts.shape[1]

    my_rank = lax.axis_index(expert_axis)
    ep = lax.axis_size(expert_axis)
    E = localE * ep

    experts_flat = selected_experts.flatten()
    sorted_indices = jnp.argsort(experts_flat)
    expert_counts = jnp.bincount(experts_flat, length=E)

    global_expert_counts: Int[Array, "R E"] = lax.all_gather(expert_counts, expert_axis)

    tokens_per_rank = int(np.ceil(K * T * config.capacity_factor))

    # compute destination rank for each token
    dests = experts_flat // localE
    metadata = _MoeMgpuMetadata(token_idx=sorted_indices, global_expert_counts=global_expert_counts)

    tokens_sorted = x.flatten()[sorted_indices]
    experts_sorted = experts_flat[sorted_indices]

    intermediate = _moe_mgpu_dispatch_w13_activation(
        tokens_sorted,
        moe_w13,
        activation_fn,
        metadata,
        config=config,
        expert_axis=expert_axis,
    )

    out = _moe_mgpu_w2_combine(
        intermediate,
        combine_weights.flatten()[sorted_indices],
        moe_w2,
        metadata,
        config=config,
        expert_axis=expert_axis,
    )

    return out


def _moe_mgpu_dispatch_w13_activation(
    tokens_sorted: Float[Array, "localTK D"],
    moe_w13: Float[Array, "localE D I2"],
    activation_fn: Callable[[jax.Array], jax.Array],
    metadata: _MoeMgpuMetadata,
    config: MoeMgpuConfig,
):
    """
    Dispatches tokens to experts, applies moe_w13 and activation_fn, and returns the intermediate representation.

    Broadly follows:
       - https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py and
       - https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py
    """
    # my_rank = lax.axis_index("expert")
    # ep = lax.axis_size("expert")

    # assert ep == 1, "Multi-rank support is not implemented yet."

    localE, D, I2 = moe_w13.shape
    I = I2 // 2

    localTK = tokens_sorted.shape[0]
    m = localTK
    i = I
    g = localE
    d = D

    assert D % config.block_k == 0
    assert I % config.block_n == 0
    block_m = config.block_m
    block_n = config.block_n
    block_k = config.block_k
    max_concurrent_steps = config.max_concurrent_steps
    grid_block_n = config.grid_block_n

    def body(rows_per_expert_gmem, lhs_gmem, rhs_gmem, o_gmem):
        grid_m = pl.cdiv(m, block_m) + g - 1
        grid_n = pl.cdiv(i, block_n)
        grid = (grid_m * grid_n,)

        @mgpu.nd_loop(grid, collective_axes="sm")
        def mn_loop(loop_info):
            mi, ni = mgpu.planar_snake(
                loop_info.index[0],
                (grid_m, grid_n),
                1,
                grid_block_n,
            )

            group_info = GroupInfo.create(
                rows_per_expert_gmem,
                block_m,
                mi,
            )

            def acc_scope(gate_acc_ref, up_acc_ref):
                def wgmma_step(_, lhs_smem, gate_smem, up_smem):
                    mgpu.wgmma(
                        gate_acc_ref,
                        lhs_smem,
                        gate_smem,
                    )
                    mgpu.wgmma(
                        up_acc_ref,
                        lhs_smem,
                        up_smem,
                    )

                mgpu.emit_pipeline(
                    wgmma_step,
                    grid=(d // block_k,),
                    in_specs=[
                        # LHS tile: [block_m, block_k]
                        mgpu.BlockSpec(
                            (block_m, block_k),
                            lambda kk: (group_info.block, kk),
                            delay_release=1,
                        ),
                        # RHS gate tile: rhs[group, kk, ni]
                        mgpu.BlockSpec(
                            (block_k, block_n),
                            lambda kk: (kk, ni),
                            delay_release=1,
                        ),
                        # RHS up tile: rhs[group, kk, ni + i / block_n]
                        mgpu.BlockSpec(
                            (block_k, block_n),
                            lambda kk: (kk, ni + i // block_n),
                            delay_release=1,
                        ),
                    ],
                    max_concurrent_steps=max_concurrent_steps,
                )(
                    lhs_gmem,
                    rhs_gmem.at[group_info.group_id],
                    rhs_gmem.at[group_info.group_id],
                )

                gate = gate_acc_ref[...]
                up = up_acc_ref[...]

                # hidden = (gate / (1.0 + exp(-gate))) * up
                hidden = activation_fn(gate) * up
                return hidden

            hidden = pl.run_scoped(
                acc_scope,
                gate_acc_ref=mgpu.ACC((block_m, block_n)),
                up_acc_ref=mgpu.ACC((block_m, block_n)),
            )

            # ridiculous log2 loop to get around lack of dynamic shapes
            @functools.partial(pl.run_scoped, o_smem=mgpu.SMEM((block_m, block_n), dtype=o_gmem.dtype))
            def store_scope(o_smem):
                o_smem[...] = hidden.astype(o_smem.dtype)
                mgpu.commit_smem()

                smem_start = group_info.start_within_block
                remaining_rows = min(block_m, m)

                while remaining_rows > 0:
                    const_rows_len = 1 << int(math.log2(remaining_rows))
                    remaining_rows //= 2

                    @pl.when(group_info.actual_size & const_rows_len != 0)
                    def _():
                        o_smem_slice = o_smem.at[pl.ds(smem_start, const_rows_len)]
                        o_gref_slice = o_gmem.at[
                            pl.ds(
                                group_info.block_start + smem_start,
                                const_rows_len,
                            ),
                            pl.ds(ni * block_n, block_n),
                        ]
                        mgpu.copy_smem_to_gmem(o_smem_slice, o_gref_slice)

                    smem_start += group_info.actual_size & const_rows_len

                mgpu.wait_smem_to_gmem(0, wait_read_only=True)

    num_sms = 132
    kernel = mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((localTK, I), tokens_sorted.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )

    local_group_sizes = metadata.global_expert_counts[0]

    return kernel(local_group_sizes, tokens_sorted, moe_w13)


def main(unused_argv):
    m, k, n, num_groups = 4096 * 8, 2560, 1280, 16
    kx, ky, kz = random.split(random.key(1234), num=3)

    lhs = jax.random.normal(kx, (m, k), jnp.bfloat16) * 0.1
    rhs = jax.random.normal(ky, (num_groups, k, n * 2), jnp.bfloat16) * 0.1
    group_boundaries = jax.lax.sort(jax.random.randint(kz, (num_groups - 1,), 0, m, jnp.int32))
    group_starts = lax.concatenate([jnp.array([0], dtype=jnp.int32), group_boundaries], 0)
    group_ends = lax.concatenate([group_boundaries, jnp.array([m], dtype=jnp.int32)], 0)
    group_sizes = group_ends - group_starts
    assert group_sizes.shape == (num_groups,)

    block_m = block_n = (64, 128, 192)
    block_k = (64,)
    max_concurrent_steps = (2, 4, 5, 6)
    grid_block_n = (1, 2, 4, 8, 16)
    configs = itertools.product(block_m, block_n, block_k, max_concurrent_steps, grid_block_n)
    names = ("block_m", "block_n", "block_k", "max_concurrent_steps", "grid_block_n")
    best_runtime = float("inf")
    best_kwargs: dict[str, int] = {}
    metadata = _MoeMgpuMetadata(
        token_idx=jnp.arange(m, dtype=jnp.int32), global_expert_counts=group_sizes.reshape(1, -1)
    )
    for config in configs:
        kwargs = dict(zip(names, config))
        if n % (kwargs["grid_block_n"] * kwargs["block_n"]):
            continue

        try:
            f = functools.partial(
                _moe_mgpu_dispatch_w13_activation, config=MoeMgpuConfig(**kwargs), activation_fn=jax.nn.silu
            )
            _, runtime = profiler.measure(f)(lhs, rhs, metadata=metadata)
        except ValueError as e:
            if "Mosaic GPU kernel exceeds available shared memory" not in str(e):
                raise
            runtime = float("inf")
        # Enable this to get more detailed information.
        else:
            assert runtime is not None
            print(" ".join(f"{k}={v}" for k, v in kwargs.items()), int(runtime * 1000))
        if runtime < best_runtime:
            best_runtime = runtime
            best_kwargs = kwargs
    if not best_kwargs:
        raise ValueError("No valid configuration found")

    def ref_ragged_dot(lhs, rhs, group_sizes):
        up = jax.lax.ragged_dot(lhs, rhs[..., n:], group_sizes=group_sizes)
        gate = jax.lax.ragged_dot(lhs, rhs[..., :n], group_sizes=group_sizes)
        return jax.nn.silu(gate) * up

    ref, ref_runtime = profiler.measure(ref_ragged_dot)(lhs, rhs, group_sizes=metadata.global_expert_counts[0])
    assert ref_runtime is not None
    result = _moe_mgpu_dispatch_w13_activation(
        lhs, rhs, metadata=metadata, config=MoeMgpuConfig(**best_kwargs), activation_fn=jax.nn.silu
    )
    np.testing.assert_allclose(result, ref, atol=1e-2, rtol=5e-2)

    tflops = float(4 * k * m * n) / (best_runtime / 1e3) / 1e12
    ref_tflops = float(4 * k * m * n) / (ref_runtime / 1e3) / 1e12
    print("Best parameters: ", " ".join(f"{k}={v}" for k, v in best_kwargs.items()))
    print(f"Kernel:    {best_runtime * 1000:.1f} us = {tflops:.1f} TFLOPS")
    print(f"Reference: {ref_runtime * 1000:.1f} us = {ref_tflops:.1f} TFLOPS")


if __name__ == "__main__":
    from absl import app

    jax.config.config_with_absl()
    app.run(main)
