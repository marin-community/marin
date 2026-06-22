# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composable parallelism for the grug-MoE pipeline: PP x FSDP x EP.

The pipeline primitive (``pipeline.py``) manualizes only the ``stage`` mesh axis
inside its ``shard_map`` and leaves every other axis *auto* (compiler-partitioned
by GSPMD). That lets a bigger slice carry two more parallelism axes *within* each
stage, with no change to the reshard-free model:

- ``data``   -- FSDP: weight matrices are sharded over ``data`` and all-gathered
  on use; the per-microbatch activation is data-parallel over the same axis.
- ``expert`` -- EP: the MoE expert axis ``E`` is sharded over ``expert`` so each
  shard owns a slice of experts; the router gather and the expert combine become
  GSPMD collectives over ``expert``.

This module builds the ``(stage, data, expert)`` mesh and the per-parameter
``PartitionSpec`` tree that places PP on the leading stage axis, FSDP on a weight
dimension, and EP on the expert dimension. The specs are structural (they name
axes, not sizes); divisibility of the sharded dimensions by the mesh axis sizes
is the caller's responsibility and fails fast in ``shard_pipeline_params``.
"""

from __future__ import annotations

import jax
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_zb.model import (
    AttentionParams,
    BlockParams,
    EmbedParams,
    HeadParams,
    MoEParams,
)
from experiments.grug.moe_zb.pipeline import STAGE_AXIS, PipelineParams

DATA_AXIS = "data"
EXPERT_AXIS = "expert"


def make_pipeline_mesh(num_stages: int, num_data: int, num_expert: int) -> Mesh:
    """Build a ``(stage, data, expert)`` mesh over the first ``S*D*E`` devices.

    All three axes are ``Auto`` so that, inside the pipeline ``shard_map`` (which
    manualizes only ``stage``), ``data``/``expert`` stay visible to GSPMD and the
    model's einsums partition + reduce over them without explicit ``out_sharding``.

    Device ordering comes from ``mesh_utils.create_device_mesh`` so the axes map to
    the physical TPU topology. Note this composes with ONE GSPMD-partitioned axis
    (PP x FSDP at ``Sx D x1`` or PP x EP at ``S x1 x E`` both lower and run on a
    v6e-8); composing TWO at once (``data`` AND ``expert`` both >1) trips XLA's SPMD
    partitioner (``spmd_partitioner_util.cc:497`` num_devices_per_group check) — it
    can't factor a GSPMD collective's device groups against a second GSPMD axis when
    a third (``stage``) is manual. Mesh axis order (stage inner/outer, topology-aware)
    does not change this; the fix is to manualize one of the two GSPMD axes (e.g. a
    hand-written expert ``psum``) so only one GSPMD axis remains under the shard_map.
    """
    need = num_stages * num_data * num_expert
    if jax.device_count() < need:
        raise ValueError(f"need {need} devices for {num_stages}x{num_data}x{num_expert}, have {jax.device_count()}")
    grid = mesh_utils.create_device_mesh((num_stages, num_data, num_expert), devices=jax.devices()[:need])
    return Mesh(grid, (STAGE_AXIS, DATA_AXIS, EXPERT_AXIS), axis_types=(AxisType.Auto,) * 3)


def _stage_spec() -> BlockParams:
    """PartitionSpec tree for the stacked per-stage block params.

    Leaves carry a leading ``[num_stages, layers_per_stage, ...]`` axis: PP on
    ``stage``, FSDP on a large weight dim over ``data``, EP on the expert dim over
    ``expert``. Norm weights and the (small) router stay replicated within a stage.
    """
    return BlockParams(
        rms_attn=P(STAGE_AXIS, None, None),
        attn=AttentionParams(
            w_q=P(STAGE_AXIS, None, None, DATA_AXIS),
            w_k=P(STAGE_AXIS, None, None, DATA_AXIS),
            w_v=P(STAGE_AXIS, None, None, DATA_AXIS),
            w_o=P(STAGE_AXIS, None, DATA_AXIS, None),
        ),
        rms_mlp=P(STAGE_AXIS, None, None),
        moe=MoEParams(
            router=P(STAGE_AXIS, None, None, None),
            w_gate=P(STAGE_AXIS, None, EXPERT_AXIS, None, DATA_AXIS),
            w_up=P(STAGE_AXIS, None, EXPERT_AXIS, None, DATA_AXIS),
            w_down=P(STAGE_AXIS, None, EXPERT_AXIS, DATA_AXIS, None),
        ),
    )


def pipeline_param_specs() -> PipelineParams:
    """The full ``PartitionSpec`` tree for ``PipelineParams`` on the 3-D mesh.

    The head's ``output_proj`` is vocab-sharded over ``stage`` (the V axis) so the
    vocab-parallel head computes the ``[D, V]`` projection once across stages, with
    the grad staying stage-sharded (no ``[D, V]`` all-reduce).
    """
    embed = EmbedParams(token_embed=P(DATA_AXIS, None), embed_norm=P())
    head = HeadParams(final_norm=P(), output_proj=P(None, STAGE_AXIS))
    return PipelineParams(embed=embed, stage=_stage_spec(), head=head)


def shard_pipeline_params(params: PipelineParams, mesh: Mesh) -> PipelineParams:
    """Reshard ``params`` onto the ``(stage, data, expert)`` mesh per the specs."""
    specs = pipeline_param_specs()
    # `params` supplies the tree structure; each PartitionSpec in `specs` is taken
    # whole as the leaf at the matching array position (flatten-up-to semantics).
    # Auto-axis meshes use device_put (not reshard, which is Explicit-axis only).
    return jax.tree_util.tree_map(
        lambda x, spec: jax.device_put(x, NamedSharding(mesh, spec)),
        params,
        specs,
    )
