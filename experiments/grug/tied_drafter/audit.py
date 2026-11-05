# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit nested architectures without allocating full model weights."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
from jax.sharding import AbstractMesh, AxisType, use_abstract_mesh

from experiments.grug.fast_track.model import Transformer
from experiments.grug.noise_belay.candidate11 import load_model_config
from experiments.grug.tied_drafter.coordinates import DraftRecipe, coordinate_map, draft_config


def linear_weight_equivalents(model: Transformer) -> float:
    """Count leading linear matmuls; excludes attention/state work and lookups."""
    total = 0.0
    for path, array in jax.tree_util.tree_flatten_with_path(model)[0]:
        name = jax.tree_util.keystr(path)
        if any(part in name for part in ("token_embed", "value_embed")):
            continue
        if "expert_mlp" in name:
            total += array.size * model.config.num_experts_per_token / model.config.num_experts
        elif array.ndim >= 2 and not any(part in name for part in (".weight", ".dt_bias", ".a_log", ".router_bias")):
            total += array.size
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_model_config()
    mesh = AbstractMesh((1, 1, 8, 1), ("replica_dcn", "data", "expert", "model"), axis_types=(AxisType.Explicit,) * 4)
    key = jax.random.PRNGKey(0)
    with use_abstract_mesh(mesh):
        hero = eqx.filter_eval_shape(Transformer.init, cfg, key=key)
        count = linear_weight_equivalents(hero)
        result = {
            "hero_parameters": sum(x.size for x in jax.tree.leaves(hero)),
            "hero_linear_weight_equivalents": count,
            "limitation": (
                "Linear matmul proxy only; no KDA recurrence, attention interactions, optimizer or transport. "
                "GPU timing is required."
            ),
            "recipes": {},
        }
        for recipe in DraftRecipe:
            config = draft_config(cfg, recipe)
            draft = eqx.filter_eval_shape(Transformer.init, config, key=key)
            coordinates = coordinate_map(hero, draft)
            result["recipes"][recipe] = {
                "parameters": sum(x.size for x in jax.tree.leaves(draft)),
                "linear_compute_fraction": linear_weight_equivalents(draft) / count,
                "hidden_dim": config.hidden_dim,
                "head_dim": config.inferred_head_dim,
                "heads": config.num_heads,
                "routed_width": config.intermediate_dim,
                "shared_width": config.shared_expert_intermediate_dim,
                "coordinates": [asdict(x) for x in coordinates],
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: {n: v for n, v in values.items() if n != "coordinates"} for k, values in result["recipes"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
