# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the native SFT parameter families against the qualified Grug optimizer groups."""

from collections import Counter

import draccus
import equinox as eqx
import jax
import numpy as np
from haliax import Axis
from jax.sharding import AxisType, Mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.utils.jax_utils import leaf_key_paths

from experiments.post_training.long_loop.exp3_recipe import exp3_optimizer

FAMILIES = {
    "expert_mlp": "muonh",
    "rms_attn": "adam",
    "rms_mlp": "adam",
    "shared": "muonh",
    "token_embed": "adam",
    "output_proj": "adamh",
    "router": "adam",
}


def optimizer_group_receipt(recipe: dict) -> dict:
    """Inspect abstract native parameters without allocating model weights or optimizer state."""
    config = draccus.decode(SnowballConfig, recipe["model"])
    axes = ("data", "context", "model", "expert", "replica", "replica_dcn")
    mesh = Mesh(np.array(jax.devices()[:1]).reshape((1,) * len(axes)), axes, axis_types=(AxisType.Explicit,) * len(axes))
    with jax.set_mesh(mesh):
        abstract = eqx.filter_eval_shape(
            SnowballLMHeadModel.init, Axis("vocab", config.vocab_size), config, key=jax.random.PRNGKey(0)
        )
    groups = exp3_optimizer(weight_decay=recipe["weight_decay"]).create_mask(abstract)
    paths = jax.tree.leaves(leaf_key_paths(abstract))
    counts = Counter()
    rows = []
    for path, param, group in zip(paths, jax.tree.leaves(abstract), jax.tree.leaves(groups), strict=True):
        for family, expected in FAMILIES.items():
            if family not in path:
                continue
            if group != expected:
                raise ValueError(
                    f"SFT optimizer family {family}: {path} {param.shape} uses {group}, expected {expected}"
                )
            counts[family] += 1
            rows.append({"parameter": path, "shape": list(param.shape), "group": group, "family": family})
    if set(counts) != FAMILIES.keys():
        raise ValueError(f"SFT optimizer admission did not inspect every family: {dict(counts)}")
    return {
        "profile": "grug_moe_hero_ep_muonh_v1",
        "family_counts": dict(counts),
        "parameters": rows,
        "allocated_model_weights": False,
    }
