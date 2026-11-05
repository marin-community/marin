# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plans and arithmetic for scoring in-memory merges of one trajectory's checkpoints.

A plan names candidates as weighted mixes of checkpoints. The eval reads each source once, newest first, and
never writes weights. It keeps the newest source and a running sum of all sources in host memory: a uniform
window or a blend of the newest with one is ``a * newest + b * running_sum`` the moment its oldest source has
been added, however many candidates share it. Any other mix gets its own host accumulator.

The router bias is a per-batch estimate rather than a learned weight. ``params.router_bias`` is the bias the
checkpoint's last step used, and ``pending_qb_betas`` sets the one the next step and an export use.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.moe_hero_ep.model import Transformer, apply_qb_betas
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig, _pin_sharding

WEIGHT_SUM_TOLERANCE = 1e-9
UNIFORM_WEIGHT_TOLERANCE = 1e-12
FIXED_NORM_OPTIMIZERS = ("muonh", "adamh")
EXPERT_BANK_PREFIX = "stacked_blocks.stacked.mlp.expert_mlp."


class BiasSource(StrEnum):
    PARAMS = "params"
    PENDING = "pending"


@dataclass(frozen=True)
class SourceWeight:
    step: int
    weight: float


@dataclass(frozen=True)
class MergeBias:
    """Replace a candidate's router bias with a weighted mix of its sources' biases."""

    source: BiasSource
    weights: tuple[SourceWeight, ...]


@dataclass(frozen=True)
class MergeCandidate:
    """One merge to score. Without ``bias``, the bias mixes linearly with the other parameters.

    With ``expert_banks_only``, only the expert banks take the mix and all else stays the newest source's.
    """

    name: str
    weights: tuple[SourceWeight, ...]
    bias: MergeBias | None = None
    reproject_to_newest: bool = False
    expert_banks_only: bool = False


@dataclass(frozen=True)
class MergePlan:
    """Candidates and raw checkpoints to score from ``step-N`` directories under ``checkpoint_root``.

    ``source_roots`` names another root for each step that a different run wrote.
    """

    checkpoint_root: str
    qk_mult: float
    score_raw_steps: tuple[int, ...]
    candidates: tuple[MergeCandidate, ...]
    source_roots: tuple[tuple[int, str], ...] = ()

    def __post_init__(self):
        names = [candidate.name for candidate in self.candidates]
        if len(set(names)) != len(names):
            raise ValueError(f"Candidate names must be distinct, got {names}")

    def source_steps(self) -> tuple[int, ...]:
        """Every checkpoint the plan reads, newest first."""
        steps = set(self.score_raw_steps)
        for candidate in self.candidates:
            steps.update(entry.step for entry in candidate.weights)
            if candidate.bias is not None:
                steps.update(entry.step for entry in candidate.bias.weights)
        return tuple(sorted(steps, reverse=True))

    def source_path(self, step: int) -> str:
        return f"{dict(self.source_roots).get(step, self.checkpoint_root)}/step-{step}"


@dataclass(frozen=True)
class RunningForm:
    """``newest * <newest source> + running * <sum of the sources through read index last_index>``."""

    newest: float
    running: float
    last_index: int


def _mix_weights(mix: Mapping[str, float], windows: Mapping[str, Sequence[int]]) -> tuple[SourceWeight, ...]:
    """Expand ``{window name: weight}`` into per-step weights, each window a uniform mean of its steps."""
    weights: dict[int, float] = {}
    for window, window_weight in mix.items():
        steps = windows[window]
        for step in steps:
            weights[step] = weights.get(step, 0.0) + window_weight / len(steps)
    total = sum(weights.values())
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"Mix {dict(mix)} has weights summing to {total}, not 1")
    return tuple(SourceWeight(step, weight) for step, weight in sorted(weights.items(), reverse=True))


def merge_plan_from_json(text: str) -> MergePlan:
    """Parse a plan whose candidates mix named windows; see ``hero_merge_plan_stage1.json``."""
    raw = json.loads(text)
    windows = raw["windows"]
    candidates = []
    for entry in raw["candidates"]:
        bias = entry.get("bias")
        candidates.append(
            MergeCandidate(
                name=entry["name"],
                weights=_mix_weights(entry["mix"], windows),
                bias=None if bias is None else MergeBias(BiasSource(bias["source"]), _mix_weights(bias["mix"], windows)),
                reproject_to_newest=entry.get("reproject_to_newest", False),
                expert_banks_only=entry.get("expert_banks_only", False),
            )
        )
    return MergePlan(
        checkpoint_root=raw["checkpoint_root"],
        qk_mult=raw["qk_mult"],
        score_raw_steps=tuple(raw["score_raw_steps"]),
        candidates=tuple(candidates),
        source_roots=tuple((int(step), root) for step, root in raw.get("source_roots", {}).items()),
    )


def running_form(weights: Sequence[SourceWeight], read_order: Sequence[int]) -> RunningForm | None:
    """The candidate as newest plus running sum, or None when its mix needs its own accumulator."""
    by_step = {entry.step: entry.weight for entry in weights}
    last_index = max(read_order.index(step) for step in by_step)
    if read_order[0] not in by_step:
        return None
    if last_index == 0:
        return RunningForm(newest=by_step[read_order[0]], running=0.0, last_index=0)
    tail = [by_step.get(read_order[index], 0.0) for index in range(1, last_index + 1)]
    if min(tail) <= 0.0 or max(tail) - min(tail) > UNIFORM_WEIGHT_TOLERANCE:
        return None
    return RunningForm(newest=by_step[read_order[0]] - tail[0], running=tail[0], last_index=last_index)


@jax.jit
def _weighted_sum(trees: tuple[Transformer, ...], coefficients: tuple[float, ...]) -> Transformer:
    def combine(*leaves):
        total = sum(coefficient * leaf for coefficient, leaf in zip(coefficients, leaves, strict=True))
        return _pin_sharding(total, leaves[0])

    return jax.tree.map(combine, *trees)


def weighted_sum(terms: Sequence[tuple[float, Transformer]]) -> Transformer:
    """``sum(coefficient * tree)``, keeping the first tree's sharding. Trees must be in device memory."""
    return _weighted_sum(tuple(tree for _, tree in terms), tuple(coefficient for coefficient, _ in terms))


def with_mixed_bias(
    params: Transformer, bias: MergeBias, biases: Mapping[int, Mapping[BiasSource, jax.Array]]
) -> Transformer:
    """Set the router bias from a mix of ``biases[step][source]``, both stored in the bias's sign convention."""
    mixed = sum(entry.weight * biases[entry.step][bias.source] for entry in bias.weights)
    return apply_qb_betas(params, -mixed)


def _fixed_norm_axes(leaf: jax.Array) -> tuple[int, ...]:
    """The axes of the Frobenius sphere MuonH and AdamH hold, under ``ExpertNormalization.ALL_EXPERTS``."""
    return tuple(range(leaf.ndim)) if leaf.ndim == 2 else tuple(range(1, leaf.ndim))


def reproject_to_reference(params: Transformer, reference: Transformer) -> Transformer:
    """Rescale each fixed-norm parameter to ``reference``'s norm, leaving Adam's parameters unchanged."""
    labels = GrugMoeMuonHConfig().create_mask(params)

    @jax.jit
    def rescale(params_in: Transformer, reference_in: Transformer) -> Transformer:
        def leaf_rescale(label: str, leaf: jax.Array, target: jax.Array) -> jax.Array:
            if label not in FIXED_NORM_OPTIMIZERS:
                return leaf
            axes = _fixed_norm_axes(leaf)
            norm = jnp.sqrt(jnp.sum(jnp.square(leaf), axis=axes, keepdims=True))
            target_norm = jnp.sqrt(jnp.sum(jnp.square(target), axis=axes, keepdims=True))
            return _pin_sharding(leaf * target_norm / jnp.maximum(norm, 1e-10), leaf)

        return jax.tree.map(leaf_rescale, labels, params_in, reference_in)

    return rescale(params, reference)


def with_expert_banks_only(params: Transformer, newest: Transformer) -> Transformer:
    """Keep ``params`` on the expert banks and take every other leaf from ``newest``."""
    names = leaf_key_paths(params)
    return jax.tree.map(
        lambda name, leaf, other: leaf if name.startswith(EXPERT_BANK_PREFIX) else other, names, params, newest
    )


def _named_leaves(tree: Transformer) -> dict[str, jax.Array]:
    return dict(zip(jax.tree.leaves(leaf_key_paths(tree)), jax.tree.leaves(tree), strict=True))


@jax.jit
def _distance_stats(source: Transformer, newest: Transformer, prior_sum: Transformer | None, prior_count: int):
    def squared(x: jax.Array, ref: jax.Array) -> jax.Array:
        # A sharded reduction over a computed intermediate over-counts unless pinned (issue #8073).
        return jnp.sum(jnp.square(_pin_sharding(x, ref)))

    newest_leaves = _named_leaves(newest)
    prior_leaves = None if prior_sum is None else _named_leaves(prior_sum)
    stats = {}
    for name, leaf in _named_leaves(source).items():
        stats[f"{name}/source_norm2"] = jnp.sum(jnp.square(leaf))
        stats[f"{name}/dist2_newest"] = squared(leaf - newest_leaves[name], leaf)
        if prior_leaves is not None:
            prior_mean = prior_leaves[name] / prior_count
            stats[f"{name}/dist2_prior_mean"] = squared(leaf - prior_mean, leaf)
            mean = (prior_leaves[name] + leaf) / (prior_count + 1)
            stats[f"{name}/mean_norm2"] = squared(mean, leaf)
            stats[f"{name}/mean_dist2_newest"] = squared(mean - newest_leaves[name], leaf)
    return stats


def distance_stats(
    source: Transformer, newest: Transformer, prior_sum: Transformer | None, prior_count: int
) -> dict[str, float]:
    """Per-parameter squared norms and distances: the Gram entries one pass can afford.

    Keys are ``<leaf path>/<stat>``. ``source_norm2`` and ``dist2_newest`` fix the source's inner product with
    the newest source. With a running sum, ``dist2_prior_mean`` is the distance from the mean of the sources
    before this one, and ``mean_norm2`` and ``mean_dist2_newest`` describe the mean including it.
    """
    return {name: float(value) for name, value in _distance_stats(source, newest, prior_sum, prior_count).items()}
