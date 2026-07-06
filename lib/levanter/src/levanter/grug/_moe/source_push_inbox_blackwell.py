# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Blackwell-specific source-push inbox staging decisions.

The Hopper inbox kernel in ``source_push_inbox`` uses Lane lowering and peer
refs directly. The Blackwell Warpgroup peer-ref probe currently fails in the
tested JAX lowering path, so the Blackwell path is staged:

1. Lane source-push copy/pack into destination-local layout.
2. Local Blackwell ragged W13/GMM using the tuned Warpgroup config below.

This module is intentionally small and side-effect free so benchmark scripts and
future production wiring can share the same architecture decision without
depending on one-off scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jax import Ref, lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.experimental.pallas.ops.gpu import blackwell_matmul_mgpu, blackwell_ragged_dot_mgpu
from jax.sharding import Mesh, PartitionSpec as P
from jaxtyping import Array, Float, Int

from levanter.grug._moe.source_push_inbox import AXIS, PushInboxConfig
from levanter.grug._moe.source_push_inbox_profiles import (
    SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072,
    SOURCE_PUSH_PROFILE_STABLE_216,
)
from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_META_LOCAL_EXPERT,
    SOURCE_PUSH_META_LOCAL_ROW_START,
    SOURCE_PUSH_META_SRC_RANK,
    SOURCE_PUSH_META_VALID_ROWS,
)


class SourcePushInboxArchitecture(StrEnum):
    """Architecture family for source-push inbox profile selection."""

    HOPPER = "hopper"
    BLACKWELL = "blackwell"


class BlackwellSourcePushStrategy(StrEnum):
    """Available Blackwell source-push implementation strategies."""

    STAGED_COPY_LOCAL_W13 = "staged_copy_local_w13"


class BlackwellPeerRefSupport(StrEnum):
    """Warpgroup peer-ref support status from the capability probe."""

    UNSUPPORTED_IN_WARPGROUP_LOWERING = "unsupported_in_warpgroup_lowering"


class BlackwellGridMinorDim(StrEnum):
    """String form of the JAX Blackwell matmul grid minor dimension."""

    M = "M"
    N = "N"


SOURCE_PUSH_BLACKWELL_W13_IMPLEMENTATION_REFERENCE = "reference"
SOURCE_PUSH_BLACKWELL_W13_IMPLEMENTATION_RAGGED = "blackwell_ragged"
SourcePushBlackwellW13Implementation: TypeAlias = Literal["reference", "blackwell_ragged"]


@dataclass(frozen=True)
class BlackwellW13TuningConfig:
    """Tuned local Blackwell ragged W13/GMM tile config."""

    tile_m: int
    tile_n: int
    tile_k: int
    max_concurrent_steps: int
    collective: bool
    grid_tile_width: int
    grid_minor_dim: BlackwellGridMinorDim
    epilogue_tile_n: int


@dataclass(frozen=True)
class BlackwellPerformanceGate:
    """Result for comparing staged source-push throughput against a baseline."""

    baseline_useful_tflops_per_rank: float
    inbox_useful_tflops_per_rank: float
    required_fraction: float = 0.60

    @property
    def required_useful_tflops_per_rank(self) -> float:
        return self.baseline_useful_tflops_per_rank * self.required_fraction

    @property
    def achieved_fraction(self) -> float:
        return self.inbox_useful_tflops_per_rank / self.baseline_useful_tflops_per_rank

    @property
    def passes(self) -> bool:
        return self.achieved_fraction >= self.required_fraction


BLACKWELL_TARGET_PROFILE = SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072
BLACKWELL_SOURCE_PUSH_STRATEGY = BlackwellSourcePushStrategy.STAGED_COPY_LOCAL_W13
BLACKWELL_PEER_REF_SUPPORT = BlackwellPeerRefSupport.UNSUPPORTED_IN_WARPGROUP_LOWERING

BLACKWELL_TARGET_W13_TUNING_CONFIG = BlackwellW13TuningConfig(
    tile_m=128,
    tile_n=128,
    tile_k=64,
    max_concurrent_steps=6,
    collective=True,
    grid_tile_width=1,
    grid_minor_dim=BlackwellGridMinorDim.M,
    epilogue_tile_n=64,
)

BLACKWELL_TARGET_W2_TUNING_CONFIG = BlackwellW13TuningConfig(
    tile_m=128,
    tile_n=64,
    tile_k=64,
    max_concurrent_steps=6,
    collective=True,
    grid_tile_width=1,
    grid_minor_dim=BlackwellGridMinorDim.M,
    epilogue_tile_n=64,
)

BLACKWELL_TARGET_W2_B200_TUNING_CONFIG = BlackwellW13TuningConfig(
    tile_m=128,
    tile_n=128,
    tile_k=64,
    max_concurrent_steps=6,
    collective=True,
    grid_tile_width=1,
    grid_minor_dim=BlackwellGridMinorDim.M,
    epilogue_tile_n=64,
)


def source_push_inbox_architecture(profile: str) -> SourcePushInboxArchitecture:
    """Return the architecture family associated with a source-push profile."""
    if profile == SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072:
        return SourcePushInboxArchitecture.BLACKWELL
    if profile == SOURCE_PUSH_PROFILE_STABLE_216:
        return SourcePushInboxArchitecture.HOPPER
    raise ValueError(
        "profile has no source-push inbox architecture mapping; "
        f"got {profile!r}, expected {SOURCE_PUSH_PROFILE_STABLE_216!r} or "
        f"{SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072!r}"
    )


def blackwell_ragged_dot_tuning_config(
    tuning_config: BlackwellW13TuningConfig = BLACKWELL_TARGET_W13_TUNING_CONFIG,
):
    """Convert the checked-in Blackwell W13 config to the upstream JAX config type."""

    return blackwell_ragged_dot_mgpu.TuningConfig(
        tile_m=tuning_config.tile_m,
        tile_n=tuning_config.tile_n,
        tile_k=tuning_config.tile_k,
        max_concurrent_steps=tuning_config.max_concurrent_steps,
        collective=tuning_config.collective,
        grid_tile_width=tuning_config.grid_tile_width,
        grid_minor_dim=blackwell_matmul_mgpu.MatmulDimension[tuning_config.grid_minor_dim.value],
        epilogue_tile_n=tuning_config.epilogue_tile_n,
    )


def source_push_reference_local_w13_h(
    destination_x: Float[Array, "Dst rows D"],
    w13: Float[Array, "Dst E D twoI"],
    group_sizes: Int[Array, "Dst E"],
) -> Float[Array, "Dst rows twoI"]:
    """Compute destination-local W13 H from rows already arranged by local expert."""

    _validate_local_w13_shapes(destination_x, w13, group_sizes)
    h = jax.vmap(
        lambda x_rank, w_rank, sizes_rank: jax.lax.ragged_dot(
            x_rank,
            w_rank,
            sizes_rank,
            preferred_element_type=jnp.float32,
        )
    )(destination_x, w13, group_sizes)
    return h.astype(w13.dtype)


def source_push_blackwell_local_w13_h(
    destination_x: Float[Array, "rows D"],
    w13: Float[Array, "E D twoI"],
    group_sizes: Int[Array, "E"],
    *,
    tuning_config: BlackwellW13TuningConfig = BLACKWELL_TARGET_W13_TUNING_CONFIG,
) -> Float[Array, "rows twoI"]:
    """Compute one destination rank's local W13 H with the Blackwell ragged kernel."""

    _validate_local_w13_rank_shapes(destination_x, w13, group_sizes)

    return blackwell_ragged_dot_mgpu.ragged_dot_kernel(
        destination_x,
        w13,
        group_sizes,
        config=blackwell_ragged_dot_tuning_config(tuning_config),
    )


def source_push_weighted_swiglu_activation(
    h: Float[Array, "Dst rows twoI"],
    h_route_weights: Float[Array, "Dst rows"],
) -> Float[Array, "Dst rows I"]:
    """Compute route-weighted SwiGLU activation from flat H rows."""

    if h.ndim != 3:
        raise ValueError(f"h must have shape [dst, rows, two_i], got {h.shape}")
    if h_route_weights.ndim != 2:
        raise ValueError(f"h_route_weights must have shape [dst, rows], got {h_route_weights.shape}")
    if h.shape[:2] != h_route_weights.shape:
        raise ValueError(f"h rows {h.shape[:2]} must match route weights {h_route_weights.shape}")
    if h.shape[-1] % 2:
        raise ValueError(f"h last dimension must be even, got {h.shape[-1]}")
    intermediate_dim = h.shape[-1] // 2
    gate = h[..., :intermediate_dim].astype(jnp.float32)
    up = h[..., intermediate_dim:].astype(jnp.float32)
    activation = jax.nn.silu(gate) * up * h_route_weights[..., None].astype(jnp.float32)
    return activation.astype(h.dtype)


def source_push_reference_local_w2_y(
    activation: Float[Array, "Dst rows I"],
    w_down: Float[Array, "Dst E I D"],
    group_sizes: Int[Array, "Dst E"],
) -> Float[Array, "Dst rows D"]:
    """Compute destination-local W2 output from flat weighted activation rows."""

    _validate_local_w13_shapes(activation, w_down, group_sizes)
    y = jax.vmap(
        lambda activation_rank, w_rank, sizes_rank: jax.lax.ragged_dot(
            activation_rank,
            w_rank,
            sizes_rank,
            preferred_element_type=jnp.float32,
        )
    )(activation, w_down, group_sizes)
    return y.astype(w_down.dtype)


def source_push_blackwell_local_w2_y(
    activation: Float[Array, "rows I"],
    w_down: Float[Array, "E I D"],
    group_sizes: Int[Array, "E"],
    *,
    tuning_config: BlackwellW13TuningConfig = BLACKWELL_TARGET_W2_TUNING_CONFIG,
) -> Float[Array, "rows D"]:
    """Compute one destination rank's W2 output with the Blackwell ragged kernel."""

    _validate_local_w13_rank_shapes(activation, w_down, group_sizes)

    return blackwell_ragged_dot_mgpu.ragged_dot_kernel(
        activation,
        w_down,
        group_sizes,
        config=blackwell_ragged_dot_tuning_config(tuning_config),
    )


def source_push_local_w13_h(
    destination_x: Float[Array, "Dst rows D"],
    w13: Float[Array, "Dst E D twoI"],
    group_sizes: Int[Array, "Dst E"],
    *,
    implementation: SourcePushBlackwellW13Implementation = SOURCE_PUSH_BLACKWELL_W13_IMPLEMENTATION_REFERENCE,
    tuning_config: BlackwellW13TuningConfig = BLACKWELL_TARGET_W13_TUNING_CONFIG,
) -> Float[Array, "Dst rows twoI"]:
    """Compute destination-local W13 H with either the reference or Blackwell local kernel."""

    if implementation == SOURCE_PUSH_BLACKWELL_W13_IMPLEMENTATION_REFERENCE:
        return source_push_reference_local_w13_h(destination_x, w13, group_sizes)
    if implementation != SOURCE_PUSH_BLACKWELL_W13_IMPLEMENTATION_RAGGED:
        raise ValueError(
            "source-push Blackwell W13 implementation must be "
            f"{SOURCE_PUSH_BLACKWELL_W13_IMPLEMENTATION_REFERENCE!r} or "
            f"{SOURCE_PUSH_BLACKWELL_W13_IMPLEMENTATION_RAGGED!r}, got {implementation!r}"
        )
    _validate_local_w13_shapes(destination_x, w13, group_sizes)
    return jax.vmap(
        lambda x_rank, w_rank, sizes_rank: source_push_blackwell_local_w13_h(
            x_rank,
            w_rank,
            sizes_rank,
            tuning_config=tuning_config,
        )
    )(destination_x, w13, group_sizes)


def _make_destination_local_x_transport_kernel(
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool = False,
):
    """Create the Lane source-push transport kernel for the staged Blackwell path."""

    config.validate()
    k_tiles = config.hidden_dim // config.block_k
    dst_offsets = tuple(range(config.ep_size))

    def body(
        packed_x_ref: Float[Ref, "Dst Q M D"],
        send_meta_ref: Int[Ref, "Dst Q F"],
        expert_base_ref: Int[Ref, "Dst E"],
        src_base_by_expert_ref: Int[Ref, "Dst S E"],
        destination_x_ref: Float[Ref, "rows D"],
    ) -> None:
        rank = lax.axis_index(AXIS)
        dst_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        k_tile = pl.program_id(2)
        k_start = k_tile * config.block_k

        def _copy_to_dst(static_dst_ordinal: int, static_dst_offset: int) -> None:
            dst = (rank + static_dst_offset) % config.ep_size
            remote_destination_x_ref = None
            if static_dst_offset != 0:
                remote_destination_x_ref = mgpu.remote_ref(
                    destination_x_ref,
                    dst,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
            valid_rows = send_meta_ref[static_dst_ordinal, entry, SOURCE_PUSH_META_VALID_ROWS]

            @pl.when(valid_rows > 0)
            def _copy_entry_tile() -> None:
                local_row_start = send_meta_ref[static_dst_ordinal, entry, SOURCE_PUSH_META_LOCAL_ROW_START]
                if use_exact_expert_major:
                    expert = send_meta_ref[static_dst_ordinal, entry, SOURCE_PUSH_META_LOCAL_EXPERT]
                    row_start = (
                        expert_base_ref[dst, expert] + src_base_by_expert_ref[dst, rank, expert] + local_row_start
                    )
                else:
                    row_start = local_row_start

                def _copy_scope(tile_smem) -> None:
                    tile_smem[:, :] = packed_x_ref[
                        static_dst_ordinal,
                        entry,
                        pl.ds(0, config.block_m),
                        pl.ds(k_start, config.block_k),
                    ]
                    mgpu.commit_smem()
                    if static_dst_offset == 0:
                        mgpu.copy_smem_to_gmem(
                            tile_smem,
                            destination_x_ref.at[
                                pl.ds(row_start, config.block_m),
                                pl.ds(k_start, config.block_k),
                            ],
                        )
                    else:
                        mgpu.copy_smem_to_gmem(
                            tile_smem,
                            remote_destination_x_ref.at[
                                pl.ds(row_start, config.block_m),
                                pl.ds(k_start, config.block_k),
                            ],
                        )
                    mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                pl.run_scoped(
                    _copy_scope,
                    tile_smem=mgpu.SMEM((config.block_m, config.block_k), dtype=packed_x_ref.dtype),
                )

        def _switch_copy_to_dst(dynamic_dst_ordinal) -> None:
            def _branch(static_dst_ordinal: int, static_dst_offset: int):
                def _copy_branch(_) -> None:
                    _copy_to_dst(static_dst_ordinal, static_dst_offset)

                return _copy_branch

            branches = tuple(
                _branch(static_dst_ordinal, static_dst_offset)
                for static_dst_ordinal, static_dst_offset in enumerate(dst_offsets)
            )
            lax.switch(dynamic_dst_ordinal, branches, None)

        _switch_copy_to_dst(dst_ordinal)

    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((config.hidden_rows_per_rank, config.hidden_dim), jnp.bfloat16),
        grid=(config.ep_size, config.entries_per_rank, k_tiles),
        grid_names=("dst_ordinal", "entry", "k_tile"),
        compiler_params=compiler_params,
    )


def sharded_destination_local_x_transport(
    mesh: Mesh,
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool = False,
):
    """Return a shard-mapped all-entry transport into `[Dst, rows, D]`."""

    kernel = _make_destination_local_x_transport_kernel(
        config,
        use_exact_expert_major=use_exact_expert_major,
    )

    def local_fn(
        packed_x_local: Float[Array, "1 Dst Q M D"],
        send_meta_local: Int[Array, "1 Dst Q F"],
        expert_base: Int[Array, "Dst E"],
        src_base_by_expert: Int[Array, "Dst S E"],
    ):
        packed_x_local = packed_x_local[0]
        send_meta_local = send_meta_local[0]
        destination_x = kernel(
            packed_x_local,
            send_meta_local,
            expert_base,
            src_base_by_expert,
        )
        return destination_x[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None, None, None),
            P(AXIS, None, None, None),
            P(None, None),
            P(None, None, None),
        ),
        out_specs=P(AXIS, None, None),
        check_vma=False,
    )


def _make_raw_destination_local_x_transport_kernel(
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool = False,
):
    """Create the staged Blackwell transport kernel that reads raw source tokens."""

    config.validate()
    k_tiles = config.hidden_dim // config.block_k
    dst_offsets = tuple(range(config.ep_size))

    def body(
        source_x_ref: Float[Ref, "T D"],
        token_ids_ref: Int[Ref, "Dst Q M"],
        send_meta_ref: Int[Ref, "Dst Q F"],
        expert_base_ref: Int[Ref, "Dst E"],
        src_base_by_expert_ref: Int[Ref, "Dst S E"],
        destination_x_ref: Float[Ref, "rows D"],
    ) -> None:
        rank = lax.axis_index(AXIS)
        dst_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        k_tile = pl.program_id(2)
        k_start = k_tile * config.block_k

        def _copy_to_dst(static_dst_ordinal: int, static_dst_offset: int) -> None:
            dst = (rank + static_dst_offset) % config.ep_size
            remote_destination_x_ref = None
            if static_dst_offset != 0:
                remote_destination_x_ref = mgpu.remote_ref(
                    destination_x_ref,
                    dst,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
            valid_rows = send_meta_ref[static_dst_ordinal, entry, SOURCE_PUSH_META_VALID_ROWS]

            @pl.when(valid_rows > 0)
            def _copy_entry_tile() -> None:
                local_row_start = send_meta_ref[static_dst_ordinal, entry, SOURCE_PUSH_META_LOCAL_ROW_START]
                if use_exact_expert_major:
                    expert = send_meta_ref[static_dst_ordinal, entry, SOURCE_PUSH_META_LOCAL_EXPERT]
                    row_start = (
                        expert_base_ref[dst, expert] + src_base_by_expert_ref[dst, rank, expert] + local_row_start
                    )
                else:
                    row_start = local_row_start

                def _copy_scope(tile_smem) -> None:
                    @pl.loop(0, config.block_m)
                    def _row_loop(row) -> None:
                        @pl.when(row < valid_rows)
                        def _copy_valid_row() -> None:
                            token = token_ids_ref[static_dst_ordinal, entry, row]
                            tile_smem[row, :] = source_x_ref[token, pl.ds(k_start, config.block_k)]

                        @pl.when(row >= valid_rows)
                        def _zero_invalid_row() -> None:
                            tile_smem[row, :] = jnp.zeros((config.block_k,), dtype=source_x_ref.dtype)

                    mgpu.commit_smem()
                    if static_dst_offset == 0:
                        mgpu.copy_smem_to_gmem(
                            tile_smem,
                            destination_x_ref.at[
                                pl.ds(row_start, config.block_m),
                                pl.ds(k_start, config.block_k),
                            ],
                        )
                    else:
                        mgpu.copy_smem_to_gmem(
                            tile_smem,
                            remote_destination_x_ref.at[
                                pl.ds(row_start, config.block_m),
                                pl.ds(k_start, config.block_k),
                            ],
                        )
                    mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                pl.run_scoped(
                    _copy_scope,
                    tile_smem=mgpu.SMEM((config.block_m, config.block_k), dtype=source_x_ref.dtype),
                )

        def _switch_copy_to_dst(dynamic_dst_ordinal) -> None:
            def _branch(static_dst_ordinal: int, static_dst_offset: int):
                def _copy_branch(_) -> None:
                    _copy_to_dst(static_dst_ordinal, static_dst_offset)

                return _copy_branch

            branches = tuple(
                _branch(static_dst_ordinal, static_dst_offset)
                for static_dst_ordinal, static_dst_offset in enumerate(dst_offsets)
            )
            lax.switch(dynamic_dst_ordinal, branches, None)

        _switch_copy_to_dst(dst_ordinal)

    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((config.hidden_rows_per_rank, config.hidden_dim), jnp.bfloat16),
        grid=(config.ep_size, config.entries_per_rank, k_tiles),
        grid_names=("dst_ordinal", "entry", "k_tile"),
        compiler_params=compiler_params,
    )


def sharded_raw_destination_local_x_transport(
    mesh: Mesh,
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool = False,
):
    """Return a shard-mapped transport from raw source-major tokens into destination rows."""

    kernel = _make_raw_destination_local_x_transport_kernel(
        config,
        use_exact_expert_major=use_exact_expert_major,
    )

    def local_fn(
        source_x_local: Float[Array, "1 T D"],
        token_ids_local: Int[Array, "1 Dst Q M"],
        send_meta_local: Int[Array, "1 Dst Q F"],
        expert_base: Int[Array, "Dst E"],
        src_base_by_expert: Int[Array, "Dst S E"],
    ):
        source_x_local = source_x_local[0]
        token_ids_local = token_ids_local[0]
        send_meta_local = send_meta_local[0]
        destination_x = kernel(
            source_x_local,
            token_ids_local,
            send_meta_local,
            expert_base,
            src_base_by_expert,
        )
        return destination_x[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None, None, None),
            P(None, None),
            P(None, None, None),
        ),
        out_specs=P(AXIS, None, None),
        check_vma=False,
    )


def sharded_blackwell_local_w13_h(
    mesh: Mesh,
    *,
    tuning_config: BlackwellW13TuningConfig = BLACKWELL_TARGET_W13_TUNING_CONFIG,
):
    """Return a shard-mapped rank-local Blackwell ragged W13 kernel."""

    def local_fn(
        destination_x_local: Float[Array, "1 rows D"],
        w13_local: Float[Array, "1 E D twoI"],
        group_sizes_local: Int[Array, "1 E"],
    ):
        h_local = source_push_blackwell_local_w13_h(
            destination_x_local[0],
            w13_local[0],
            group_sizes_local[0],
            tuning_config=tuning_config,
        )
        return h_local[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None),
        ),
        out_specs=P(AXIS, None, None),
        check_vma=False,
    )


def sharded_blackwell_local_w2_y(
    mesh: Mesh,
    *,
    tuning_config: BlackwellW13TuningConfig | None = None,
):
    """Return a shard-mapped rank-local weighted SwiGLU plus Blackwell W2 kernel."""
    if tuning_config is None:
        tuning_config = _default_blackwell_w2_tuning_config(mesh)

    def local_fn(
        h_local: Float[Array, "1 rows twoI"],
        h_route_weights_local: Float[Array, "1 rows"],
        w_down_local: Float[Array, "1 E I D"],
        group_sizes_local: Int[Array, "1 E"],
    ):
        activation_local = source_push_weighted_swiglu_activation(h_local, h_route_weights_local)[0]
        y_local = source_push_blackwell_local_w2_y(
            activation_local,
            w_down_local[0],
            group_sizes_local[0],
            tuning_config=tuning_config,
        )
        return y_local[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None),
            P(AXIS, None),
            P(AXIS, None, None, None),
            P(AXIS, None),
        ),
        out_specs=P(AXIS, None, None),
        check_vma=False,
    )


def _default_blackwell_w2_tuning_config(mesh: Mesh) -> BlackwellW13TuningConfig:
    device_kind = getattr(mesh.devices.flat[0], "device_kind", "")
    if "B200" in device_kind:
        return BLACKWELL_TARGET_W2_B200_TUNING_CONFIG
    return BLACKWELL_TARGET_W2_TUNING_CONFIG


def _make_flat_y_return_transport_kernel(
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool = False,
):
    """Create the Lane return transport from flat destination-local W2 rows."""

    config.validate()
    n_tiles = config.hidden_dim // config.block_n
    src_offsets = tuple(range(config.ep_size))

    def body(
        y_ref: Float[Ref, "rows D"],
        recv_meta_ref: Int[Ref, "SRC Q F"],
        expert_base_ref: Int[Ref, "E"],
        src_base_by_expert_ref: Int[Ref, "SRC E"],
        source_return_ref: Float[Ref, "DST Q M D"],
    ) -> None:
        rank = lax.axis_index(AXIS)
        src_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        n_tile = pl.program_id(2)
        n_start = n_tile * config.block_n

        def _copy_to_src(static_src_ordinal: int) -> None:
            src = (rank + static_src_ordinal) % config.ep_size
            dst_ordinal = (-static_src_ordinal) % config.ep_size
            remote_source_return_ref = None
            if static_src_ordinal != 0:
                remote_source_return_ref = mgpu.remote_ref(
                    source_return_ref,
                    src,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
            valid_rows = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_VALID_ROWS]

            @pl.when(valid_rows > 0)
            def _copy_entry_tile() -> None:
                local_row_start = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_LOCAL_ROW_START]
                if use_exact_expert_major:
                    src_rank = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_SRC_RANK]
                    expert = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_LOCAL_EXPERT]
                    row_start = expert_base_ref[expert] + src_base_by_expert_ref[src_rank, expert] + local_row_start
                else:
                    row_start = local_row_start

                def _copy_scope(tile_smem) -> None:
                    tile_smem[:, :] = y_ref[
                        pl.ds(row_start, config.block_m),
                        pl.ds(n_start, config.block_n),
                    ]
                    mgpu.commit_smem()
                    if static_src_ordinal == 0:
                        mgpu.copy_smem_to_gmem(
                            tile_smem,
                            source_return_ref.at[
                                dst_ordinal,
                                entry,
                                pl.ds(0, config.block_m),
                                pl.ds(n_start, config.block_n),
                            ],
                        )
                    else:
                        mgpu.copy_smem_to_gmem(
                            tile_smem,
                            remote_source_return_ref.at[
                                dst_ordinal,
                                entry,
                                pl.ds(0, config.block_m),
                                pl.ds(n_start, config.block_n),
                            ],
                        )
                    mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                pl.run_scoped(
                    _copy_scope,
                    tile_smem=mgpu.SMEM((config.block_m, config.block_n), dtype=y_ref.dtype),
                )

        def _switch_copy_to_src(dynamic_src_ordinal) -> None:
            def _branch(static_src_ordinal: int):
                def _copy_branch(_) -> None:
                    _copy_to_src(static_src_ordinal)

                return _copy_branch

            branches = tuple(_branch(static_src_ordinal) for static_src_ordinal in src_offsets)
            lax.switch(dynamic_src_ordinal, branches, None)

        _switch_copy_to_src(src_ordinal)

    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct(
            (config.ep_size, config.entries_per_rank, config.block_m, config.hidden_dim),
            jnp.bfloat16,
        ),
        grid=(config.ep_size, config.entries_per_rank, n_tiles),
        grid_names=("src_ordinal", "entry", "n_tile"),
        compiler_params=compiler_params,
    )


def sharded_flat_y_return_transport(
    mesh: Mesh,
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool = False,
):
    """Return a shard-mapped Lane transport from flat W2 rows to source queues."""

    kernel = _make_flat_y_return_transport_kernel(
        config,
        use_exact_expert_major=use_exact_expert_major,
    )

    def local_fn(
        y_local: Float[Array, "1 rows D"],
        recv_meta_local: Int[Array, "1 SRC Q F"],
        expert_base_local: Int[Array, "1 E"],
        src_base_by_expert_local: Int[Array, "1 SRC E"],
    ):
        source_return = kernel(
            y_local[0],
            recv_meta_local[0],
            expert_base_local[0],
            src_base_by_expert_local[0],
        )
        return source_return[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None),
            P(AXIS, None, None),
        ),
        out_specs=P(AXIS, None, None, None, None),
        check_vma=False,
    )


def _validate_local_w13_shapes(
    destination_x: jax.Array,
    w13: jax.Array,
    group_sizes: jax.Array,
) -> None:
    if destination_x.ndim != 3:
        raise ValueError(f"destination_x must have shape [dst, rows, hidden], got {destination_x.shape}")
    if w13.ndim != 4:
        raise ValueError(f"w13 must have shape [dst, expert, hidden, two_i], got {w13.shape}")
    if group_sizes.ndim != 2:
        raise ValueError(f"group_sizes must have shape [dst, expert], got {group_sizes.shape}")
    if destination_x.shape[0] != w13.shape[0] or destination_x.shape[0] != group_sizes.shape[0]:
        raise ValueError(
            "destination_x, w13, and group_sizes must have the same destination-rank dimension; "
            f"got {destination_x.shape[0]}, {w13.shape[0]}, {group_sizes.shape[0]}"
        )
    _validate_local_w13_rank_shapes(destination_x[0], w13[0], group_sizes[0])


def _validate_local_w13_rank_shapes(
    destination_x: jax.Array,
    w13: jax.Array,
    group_sizes: jax.Array,
) -> None:
    if destination_x.ndim != 2:
        raise ValueError(f"destination_x rank-local input must have shape [rows, hidden], got {destination_x.shape}")
    if w13.ndim != 3:
        raise ValueError(f"w13 rank-local input must have shape [expert, hidden, two_i], got {w13.shape}")
    if group_sizes.ndim != 1:
        raise ValueError(f"group_sizes rank-local input must have shape [expert], got {group_sizes.shape}")
    if destination_x.shape[-1] != w13.shape[-2]:
        raise ValueError(f"destination_x hidden dim {destination_x.shape[-1]} must match w13 dim {w13.shape[-2]}")
    if w13.shape[0] != group_sizes.shape[0]:
        raise ValueError(f"w13 expert dim {w13.shape[0]} must match group_sizes dim {group_sizes.shape[0]}")


def blackwell_performance_gate(
    *,
    baseline_useful_tflops_per_rank: float,
    inbox_useful_tflops_per_rank: float,
    required_fraction: float = 0.60,
) -> BlackwellPerformanceGate:
    """Compare staged inbox throughput against a local Blackwell compute baseline."""
    if baseline_useful_tflops_per_rank <= 0:
        raise ValueError(f"baseline_useful_tflops_per_rank must be positive, got {baseline_useful_tflops_per_rank}")
    if inbox_useful_tflops_per_rank < 0:
        raise ValueError(f"inbox_useful_tflops_per_rank must be non-negative, got {inbox_useful_tflops_per_rank}")
    if not 0 < required_fraction <= 1:
        raise ValueError(f"required_fraction must be in (0, 1], got {required_fraction}")
    return BlackwellPerformanceGate(
        baseline_useful_tflops_per_rank=baseline_useful_tflops_per_rank,
        inbox_useful_tflops_per_rank=inbox_useful_tflops_per_rank,
        required_fraction=required_fraction,
    )
