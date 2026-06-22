# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Documented blocker: gradient parity (Step 2) of the pipelined production model.

The pipelined FORWARD lowers and matches the oracle exactly (see
``check_forward.py``). Differentiating it with ``jax.value_and_grad`` does NOT
work, for a structural reason rooted in how the production model shards:

- The production model reshards activations with EXPLICIT-axis specs throughout
  (``out_sharding=`` on the embed gather / einsums, ``reshard(..., _batch_spec())``
  in every Block, the fused-CE head's internal shard_map). Those calls require the
  surviving mesh axes (``data``/``expert``/``model``) to be ``AxisType.Explicit``
  inside the outer ``stage``-manual ``shard_map``.

- A ``shard_map`` keeps its non-manual axes Explicit only with ``check_vma=False``.
  But JAX cannot transpose (autodiff) a ``check_vma=False`` ``shard_map`` whose body
  reshards over those axes: linearization routes through ``_unmatch_spec`` and
  rejects out_specs that name non-manual axes.

- With the default ``check_vma=True`` the surviving axes become ``AxisType.Auto``
  inside the body, and the model's first Explicit reshard (the embed gather) raises
  ``context mesh ... should match the aval mesh ... (Manual, Auto, Auto, Auto, Auto)``.

So Step 2 as specified -- ``jax.value_and_grad`` of the pipelined loss vs the
oracle -- is blocked by the production model's in-body Explicit reshards. The toy
``moe_zb`` model autodiffs its pipeline because it is reshard-free (it relies on
GSPMD inference, never calling ``reshard``/``out_sharding`` in the body), keeping
``check_vma=True`` valid. Closing Step 2 for the production model requires making
the model's inter-layer activation sharding inference-driven (or otherwise
shard_map-autodiff compatible) rather than Explicit-reshard-driven.

This script triggers the blocker and prints the exact error, so the failure mode
is reproducible.

Run:

    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        uv run python -m experiments.grug.moe_pp.check_grad_blocked
"""

from __future__ import annotations

import logging
import os

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import Transformer
from experiments.grug.moe_pp.check_forward import CONFIG, STAGE
from experiments.grug.moe_pp.pipeline import pipeline_forward_loss, stack_blocks_for_stages

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mesh = compact_grug_mesh(expert_axis_size=2, replica_axis_size=1, model_axis_size=1, stage_axis_size=STAGE)
    key = jax.random.PRNGKey(0)
    k_model, k_tokens = jax.random.split(key, 2)
    with set_mesh(mesh):
        model = Transformer.init(CONFIG, key=k_model)
        tokens = jax.random.randint(k_tokens, (16, 64), 0, CONFIG.vocab_size, dtype=jnp.int32)
        weight = jnp.ones((16, 64), dtype=jnp.float32)
        stage_arrays, block_static = stack_blocks_for_stages(model, STAGE)
        model_arrays = eqx.filter(model, eqx.is_array)
        _, static = eqx.partition(model, eqx.is_array)

        def loss_fn(arrays, stage_arrays):
            m = eqx.combine(arrays, static)
            return pipeline_forward_loss(m, stage_arrays, block_static, tokens, weight, mesh=mesh, num_stages=STAGE)

        try:
            jax.value_and_grad(loss_fn, argnums=(0, 1))(model_arrays, stage_arrays)
            logger.info("UNEXPECTED: value_and_grad succeeded")
            return 0
        except ValueError as ex:
            logger.info("EXPECTED BLOCKER (%s): %s", type(ex).__name__, str(ex).splitlines()[0])
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
