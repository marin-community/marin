# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit nested coordinates for Belay draft proposals."""

from dataclasses import dataclass, replace
from enum import StrEnum

import jax

from experiments.grug.fast_track.model import GrugModelConfig, Transformer


class DraftRecipe(StrEnum):
    FFN_EIGHTH = "ffn_eighth"
    WIDTH_QUARTER = "width_quarter"
    WIDTH_EIGHTH = "width_eighth"


@dataclass(frozen=True)
class Coordinate:
    path: str
    hero_shape: tuple[int, ...]
    draft_shape: tuple[int, ...]
    propose: bool


def draft_config(hero: GrugModelConfig, recipe: DraftRecipe) -> GrugModelConfig:
    """Choose nested widths while retaining layers, vocabulary and expert identities."""
    if recipe == DraftRecipe.FFN_EIGHTH:
        return replace(
            hero,
            intermediate_dim=hero.intermediate_dim // 8,
            shared_expert_intermediate_dim=hero.shared_expert_intermediate_dim // 8,
        )
    divisor = 4 if recipe == DraftRecipe.WIDTH_QUARTER else 8
    width = hero.hidden_dim // divisor
    # Retain the first original attention head, narrowing it only when necessary.
    # Prefix slicing then has an explicit head-coordinate interpretation.
    return replace(
        hero,
        hidden_dim=width,
        head_dim=min(hero.inferred_head_dim, width),
        num_heads=1,
        global_local_heads=1,
        latent_dim=hero.latent_dim // divisor,
        mla_kv_latent_dim=hero.mla_kv_latent_dim // divisor,
        intermediate_dim=hero.intermediate_dim // 8,
        shared_expert_intermediate_dim=hero.shared_expert_intermediate_dim // 8,
    )


def coordinate_map(hero: Transformer, draft: Transformer) -> tuple[Coordinate, ...]:
    """Validate parameter identity and record a prefix selection for each array."""
    hero_paths, _ = jax.tree_util.tree_flatten_with_path(hero)
    draft_paths, _ = jax.tree_util.tree_flatten_with_path(draft)
    coordinates = []
    for (hp, hv), (dp, dv) in zip(hero_paths, draft_paths, strict=True):
        path = jax.tree_util.keystr(hp)
        if path != jax.tree_util.keystr(dp):
            raise ValueError(f"Parameter identity changed at {path}")
        if len(hv.shape) != len(dv.shape) or any(d > h for h, d in zip(hv.shape, dv.shape, strict=True)):
            raise ValueError(f"Draft coordinates do not nest at {path}: {hv.shape}, {dv.shape}")
        coordinates.append(Coordinate(path, hv.shape, dv.shape, not path.endswith(".router_bias")))
    return tuple(coordinates)


def restrict_model(hero: Transformer, draft_template: Transformer, coordinates: tuple[Coordinate, ...]) -> Transformer:
    """Read selected hero weights into the draft's architecture and tree structure."""
    hero_leaves = jax.tree.leaves(hero)
    _, draft_tree = jax.tree.flatten(draft_template)
    selected = [
        leaf[tuple(slice(0, size) for size in coordinate.draft_shape)]
        for leaf, coordinate in zip(hero_leaves, coordinates, strict=True)
    ]
    return jax.tree.unflatten(draft_tree, selected)


def lift_displacement(
    current: Transformer,
    draft_start: Transformer,
    draft_end: Transformer,
    coordinates: tuple[Coordinate, ...],
    *,
    scale: float,
) -> Transformer:
    """Add a stale draft displacement to current hero weights without mutating them.

    Router biases belong to the hero's adaptive load-balancing state and are
    retained. All omitted coordinates retain their current hero values.
    """
    current_leaves, current_tree = jax.tree.flatten(current)
    start_leaves = jax.tree.leaves(draft_start)
    end_leaves = jax.tree.leaves(draft_end)
    result = []
    for hero, before, after, coordinate in zip(current_leaves, start_leaves, end_leaves, coordinates, strict=True):
        if not coordinate.propose:
            result.append(hero)
            continue
        selection = tuple(slice(0, size) for size in coordinate.draft_shape)
        update = hero[selection] + scale * (after.astype(hero.dtype) - before.astype(hero.dtype))
        result.append(jax.lax.dynamic_update_slice(hero, update, (0,) * hero.ndim) if hero.ndim else update)
    return jax.tree.unflatten(current_tree, result)
