# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""cuDNN Frontend grouped weight gradients through the CuTeDSL JAX bridge."""

import functools
import importlib
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call

# Row count each expert group is padded up to, so every group starts where the kernel's
# tile loads expect it to.
#
# This MUST match the kernel's own `FIX_PAD_SIZE`. cuDNN Frontend's grouped-Wgrad contract
# requires every expert's token count to be divisible by this, and the kernel does not check:
# it derives its per-expert tile count as `ceil(tokens_i / cta_tile_k)` while addressing through
# one TMA descriptor over the whole buffer, so a group whose row count is not a multiple of the
# tile reads its successor's rows as if they were its own and silently returns a wrong weight
# gradient. `_assert_group_alignment_matches_kernel` pins this to the installed kernel so the
# value cannot drift again (it was 8 here against a 256-row contract, which corrupted dw13 and
# dw2 for every expert but the last in each call -- see issue #8339).
_GROUP_ALIGNMENT = 256
# Vector width the feature (non-grouped) dimensions must divide, which is what the tensor
# specs below declare to the kernel.
_FEATURE_ALIGNMENT = 8
_MMA_TILER_MN = (256, 256)
_CLUSTER_SHAPE_MN = (2, 1)


class _CudnnModules(NamedTuple):
    """CuTeDSL and cuDNN Frontend entry points, imported only where the GPU extra exists."""

    cutlass: Any
    cute: Any
    cutlass_jax: Any
    kernel_type: Any
    weight_mode: Any
    input_order: Any


@functools.cache
def _cudnn_modules() -> _CudnnModules:
    kernel_module = importlib.import_module("cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad")
    utility_module = importlib.import_module("cudnn.gemm.cutedsl.grouped.moe_utils")
    return _CudnnModules(
        cutlass=importlib.import_module("cutlass"),
        cute=importlib.import_module("cutlass.cute"),
        cutlass_jax=importlib.import_module("cutlass.jax"),
        kernel_type=kernel_module.MoEGroupedGemmWgradBF16Kernel,
        weight_mode=utility_module.MoEWeightMode,
        input_order=utility_module.WGradInputOrder,
    )


def _assert_group_alignment_matches_kernel() -> None:
    """Fail loudly if the installed kernel's required padding differs from `_GROUP_ALIGNMENT`.

    The kernel silently returns wrong gradients when its groups are misaligned, so a mismatch
    here has to raise rather than degrade.
    """
    required = getattr(_cudnn_modules().kernel_type, "FIX_PAD_SIZE", None)
    if required is not None and required != _GROUP_ALIGNMENT:
        raise RuntimeError(
            f"cuDNN grouped Wgrad pads groups to {_GROUP_ALIGNMENT} rows but the installed kernel "
            f"requires {required}. Misaligned groups are computed silently wrong, so refusing to run."
        )


@cute_launcher_factory
def _build_launcher(
    modules,
    *,
    expert_count: int,
    max_active_clusters: int,
):
    @modules.cute.jit
    def launcher(stream, mat_a, mat_b, offsets, output, workspace):
        kernel = modules.kernel_type(
            acc_dtype=modules.cutlass.Float32,
            use_2cta_instrs=True,
            mma_tiler_mn=_MMA_TILER_MN,
            cluster_shape_mn=_CLUSTER_SHAPE_MN,
            accumulate_on_output=False,
            expert_cnt=expert_count,
            weight_mode=modules.weight_mode.DENSE,
            input_order=modules.input_order.Tensor2D,
        )
        kernel(mat_a, mat_b, output, offsets, workspace, max_active_clusters, stream, None)

    return launcher


def aligned_group_capacity(rows: int, groups: int, alignment: int = _GROUP_ALIGNMENT) -> int:
    """Rows a buffer needs so `groups` groups holding `rows` payload rows partition it aligned.

    Rounding each group's extent up to ``alignment`` can cost at most one extra alignment per
    group beyond rounding the total up once, and that worst case is never reached by every group
    at once, so ``groups - 1`` alignments of slack is both sufficient and tight. The result is
    itself a multiple of ``alignment``, which is what lets the last group absorb the leftover and
    still divide it.

    ``alignment == 1`` is the unaligned layout, where extents are the payload counts and no slack
    is needed at all.
    """
    if alignment < 1:
        raise ValueError(f"alignment must be positive, got {alignment}")
    if alignment == 1:
        return rows
    return -(-rows // alignment) * alignment + alignment * (groups - 1)


def full_partition_offsets(rows: int, leading_group_sizes: jax.Array) -> jax.Array:
    """Per-group end rows that partition all ``rows``, giving the last group the leftover.

    Takes the first ``groups - 1`` extents, because the last one is not a free choice: the
    kernel's contract is not only that every extent divides `_GROUP_ALIGNMENT`, but that the final
    offset equals the operand's physical row count, since the kernel walks its groups across one
    TMA descriptor over the whole buffer. So the last group gets every remaining row, and a
    signature that cannot be handed a different value is the honest one.

    Those leftover rows contribute nothing to the last group's product as long as they are zero in
    both operands, which is what every caller here guarantees.
    """
    leading_group_sizes = jnp.asarray(leading_group_sizes, dtype=jnp.int32)
    tail_size = rows - jnp.sum(leading_group_sizes, dtype=jnp.int32)
    return jnp.cumsum(jnp.concatenate([leading_group_sizes, tail_size[None]]), dtype=jnp.int32)


def pad_grouped_rows(values: jax.Array, group_sizes: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Insert zero rows so each contiguous expert group ends on a `_GROUP_ALIGNMENT` boundary."""
    if values.ndim != 2:
        raise ValueError(f"values must be rank 2, got shape={values.shape}")
    if values.shape[0] == 0:
        raise ValueError("values must contain at least one physical row")
    if group_sizes.ndim != 1 or group_sizes.shape[0] == 0:
        raise ValueError(f"group_sizes must be a nonempty vector, got shape={group_sizes.shape}")

    group_sizes = group_sizes.astype(jnp.int32)
    groups = group_sizes.shape[0]

    # The kernel's contract has two halves: every group extent divides `_GROUP_ALIGNMENT`, AND the
    # final offset equals the buffer's physical row count. Rounding each extent up satisfies only
    # the first and leaves a ragged tail outside the last group, so give the last group every
    # remaining row instead. Those extra rows are zero and contribute nothing to its product.
    #
    # The physical size is static (it must not depend on runtime routing): rounding the capacity up
    # and adding one alignment per preceding group is always enough for the rounded extents.
    padded_capacity = aligned_group_capacity(values.shape[0], groups)

    head_sizes = ((group_sizes[:-1] + _GROUP_ALIGNMENT - 1) // _GROUP_ALIGNMENT) * _GROUP_ALIGNMENT
    active_offsets = jnp.cumsum(group_sizes, dtype=jnp.int32)
    padded_offsets = full_partition_offsets(padded_capacity, head_sizes)
    active_starts = jnp.concatenate([jnp.zeros((1,), jnp.int32), active_offsets[:-1]])
    padded_starts = jnp.concatenate([jnp.zeros((1,), jnp.int32), padded_offsets[:-1]])

    padded_rows = jnp.arange(padded_capacity, dtype=jnp.int32)
    expert_ids = jnp.sum(padded_rows[:, None] >= padded_offsets[None, :], axis=1, dtype=jnp.int32)
    safe_expert_ids = jnp.minimum(expert_ids, group_sizes.shape[0] - 1)
    rows_within_group = padded_rows - padded_starts[safe_expert_ids]
    source_rows = active_starts[safe_expert_ids] + rows_within_group
    valid = (expert_ids < group_sizes.shape[0]) & (rows_within_group < group_sizes[safe_expert_ids])
    source_rows = jnp.clip(source_rows, 0, values.shape[0] - 1)
    padded = jnp.where(valid[:, None], values[source_rows], jnp.zeros((), dtype=values.dtype))
    return padded, padded_offsets


def cudnn_grouped_wgrad(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """Compute per-expert ``lhs.T @ rhs`` for contiguous ragged row groups.

    Both operands are copied into the kernel's aligned layout first, which is what
    `pad_grouped_rows` costs. A caller whose buffers already carry that layout should use
    `cudnn_grouped_wgrad_prealigned`.
    """
    _assert_group_alignment_matches_kernel()
    lhs_padded, padded_offsets = pad_grouped_rows(lhs, group_sizes)
    rhs_padded, _ = pad_grouped_rows(rhs, group_sizes)
    return _grouped_wgrad(lhs_padded, rhs_padded, padded_offsets, expert_count=group_sizes.shape[0])


def cudnn_grouped_wgrad_prealigned(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """`cudnn_grouped_wgrad` for operands already in the kernel's aligned layout.

    The caller guarantees that group ``i`` occupies rows
    ``[sum(group_sizes[:i]), sum(group_sizes[:i+1]))`` of both operands, that every entry of
    ``group_sizes`` is a multiple of `_GROUP_ALIGNMENT` (so every group starts on an aligned
    row), and that every row of both operands not covered by an arrival is zero.

    ``group_sizes[-1]`` is ignored: rows past the last group are folded into it by
    `full_partition_offsets`, because the kernel requires the final offset to equal the operand's
    row count -- it addresses every group through one descriptor over the whole buffer, so a group
    that ends short reads on into whatever follows. That leftover is the buffer's own slack, which
    the caller has zeroed, so it contributes nothing.

    Under that contract this is `cudnn_grouped_wgrad` with the pad pass removed: `pad_grouped_rows`
    applied to such a layout reproduces it row for row and returns the same offsets, so the kernel
    sees byte-identical operands and offsets either way.
    """
    _assert_group_alignment_matches_kernel()
    offsets = full_partition_offsets(lhs.shape[0], group_sizes[:-1])
    return _grouped_wgrad(lhs, rhs, offsets, expert_count=group_sizes.shape[0])


def _grouped_wgrad(lhs: jax.Array, rhs: jax.Array, offsets: jax.Array, *, expert_count: int) -> jax.Array:
    """Call the kernel over row groups delimited by ``offsets``, the per-group end rows."""
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError(f"lhs and rhs must be rank 2, got lhs={lhs.shape}, rhs={rhs.shape}")
    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError(f"lhs and rhs row counts must match, got lhs={lhs.shape}, rhs={rhs.shape}")
    if lhs.dtype != jnp.bfloat16 or rhs.dtype != jnp.bfloat16:
        raise ValueError(f"cuDNN grouped Wgrad requires BF16 inputs, got lhs={lhs.dtype}, rhs={rhs.dtype}")
    if lhs.shape[1] % _FEATURE_ALIGNMENT != 0 or rhs.shape[1] % _FEATURE_ALIGNMENT != 0:
        raise ValueError(
            f"cuDNN grouped Wgrad feature dimensions must divide {_FEATURE_ALIGNMENT}, "
            f"got lhs={lhs.shape}, rhs={rhs.shape}"
        )

    modules = _cudnn_modules()
    max_active_clusters = modules.cutlass.utils.HardwareInfo().get_max_active_clusters(
        _CLUSTER_SHAPE_MN[0] * _CLUSTER_SHAPE_MN[1]
    )
    launcher = _build_launcher(
        modules,
        expert_count=expert_count,
        max_active_clusters=max_active_clusters,
    )
    tensor_spec = modules.cutlass_jax.TensorSpec
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((expert_count, lhs.shape[1], rhs.shape[1]), lhs.dtype),
            jax.ShapeDtypeStruct((1,), jnp.uint8),
        ),
        input_spec=(
            tensor_spec(mode=(1, 0), divisibility=(1, _FEATURE_ALIGNMENT), static=False),
            tensor_spec(divisibility=(1, _FEATURE_ALIGNMENT), static=False),
            tensor_spec(static=False),
        ),
        output_spec=(
            tensor_spec(divisibility=(1, 1, _FEATURE_ALIGNMENT), static=False),
            tensor_spec(static=False),
        ),
        use_static_tensors=False,
    )
    output, _workspace = call(lhs, rhs, offsets)
    return output
