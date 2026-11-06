# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.sharding import compact_grug_mesh

from experiments.evaluation.grug_parity.reference import restore_hero_bf16
from experiments.grug.moe_hero_ep.ops.test_export_vllm import native_fixture


def test_reference_restores_master_weights_and_pending_router_update(tmp_path):
    request, original = native_fixture(str(tmp_path / "checkpoint"), master=True)
    manifest = {
        "checkpoint": {"uri": request.checkpoint, "metadata_digest": request.metadata_digest},
        "source_revision": request.source_revision,
    }
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        restored = restore_hero_bf16(manifest, request.model, mesh)
        expected = jax.tree.map(lambda value: np.asarray(value).astype(jnp.bfloat16), original)
        # The fixture stores bias 9 in both saved models. Pending QB replaces it.
        bias = np.asarray([[-0.5, 2.5, -3.5, 1.5], [5.5, 0.5, -0.5, -5.5]], dtype=jnp.bfloat16)
        for actual, wanted in zip(jax.tree.leaves(restored), jax.tree.leaves(expected), strict=True):
            if actual is restored.stacked_blocks.stacked.mlp.router_bias:
                wanted = bias
            np.testing.assert_array_equal(np.asarray(actual), wanted)
            assert actual.dtype == jnp.bfloat16
