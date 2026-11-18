# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deletion Tuning for the Levanter Snowball model (arXiv 2611.10866).

One trainable logit per masked expert weight; the pretrained weights stay
frozen and the forward reads ``W * sigmoid(alpha * G)`` (soft) during training
and ``W * 1[G > 0]`` (hard) for evaluation and export. Gate logits start from
Wanda scores captured on a calibration pass over the run's training data, with
the global delete-group threshold fixed from a strided host-pooled score
subsample (sorting a sharded tensor is not expressible). Levanter's
``TrainerState.init(is_trainable=...)`` natively scopes optimizer state,
gradients, and checkpointing to the gate logits when given this module's
filter.
"""

import functools
import math
from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

DeletionTuningDefaults = dict(
    eta=1 / 2048,
    keep_logit_mean=0.03,
    delete_logit_mean=-0.03,
    logit_std=0.01,
    lam=1e-5,
    alpha_start_sft=1e2,
    alpha_end_sft=1e3,
)

WANDA_SAMPLE_STRIDE = 64
"""Output-axis stride for the Wanda score subsample that fixes the delete threshold."""


class DeletionTuningScope(StrEnum):
    ROUTED_GATE = "routed_gate"
    ROUTED_AND_SHARED_ALL = "routed_and_shared_all"


@dataclass(frozen=True)
class DeletionTuningConfig:
    """Deletion Tuning settings carried on ``SnowballConfig``."""

    scope: DeletionTuningScope = DeletionTuningScope.ROUTED_GATE
    eta: float = DeletionTuningDefaults["eta"]
    lam: float = DeletionTuningDefaults["lam"]
    alpha_start: float = DeletionTuningDefaults["alpha_start_sft"]
    alpha_end: float = DeletionTuningDefaults["alpha_end_sft"]
    lr: float = 5e-4
    calibration_sequences: int = 2048
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.eta < 1.0:
            raise ValueError(f"eta must be in (0, 1), got {self.eta}")
        if self.lam < 0:
            raise ValueError(f"lam must be non-negative, got {self.lam}")
        if self.alpha_start <= 0 or self.alpha_end <= self.alpha_start:
            raise ValueError(f"alpha must ramp upward: start={self.alpha_start}, end={self.alpha_end}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")

    @property
    def masks_up_and_down(self) -> bool:
        return self.scope == DeletionTuningScope.ROUTED_AND_SHARED_ALL

    @property
    def masks_shared_experts(self) -> bool:
        return self.scope == DeletionTuningScope.ROUTED_AND_SHARED_ALL


def deletion_mask(logits: jax.Array, weight: jax.Array, alpha: jax.Array, hard: jax.Array) -> jax.Array:
    """The masked weight ``stop_gradient(W) * sigmoid(alpha * G)`` (or ``1[G>0]``).

    A scalar ``jnp.where`` would broadcast a replicated condition against
    sharded cases, which explicit sharding rejects; the hard/soft switch mixes
    arithmetically instead.
    """
    soft = jax.nn.sigmoid(jax.lax.stop_gradient(alpha).astype(jnp.float32) * logits.astype(jnp.float32))
    hard_mask = (logits > 0).astype(soft.dtype)
    hard_f = hard.astype(soft.dtype)
    mask = soft * (1.0 - hard_f) + hard_mask * hard_f
    return jax.lax.stop_gradient(weight) * mask


def deletion_trainable_filter(model) -> bool:
    """A filter marking the gate-logit leaves for ``TrainerState.init(is_trainable=...)``.

    Levanter's filter spec receives leaves; a path-based tree of bools built
    over the model structure marks exactly the ``*_logits`` fields.
    """
    leaves, treedef = jax.tree_util.tree_flatten_with_path(model)
    keep = []
    for path, _leaf in leaves:
        key = getattr(path[-1], "name", None) or getattr(path[-1], "key", None)
        keep.append(isinstance(key, str) and key.endswith("_logits"))
    return treedef.unflatten(keep)


def _tree_replace(where, model, value):
    """``eqx.tree_at`` that can replace ``None`` leaves (the uninitialized logits)."""
    return eqx.tree_at(where, model, value, is_leaf=lambda x: x is None)


def _local_wanda_sums(x, selected_experts, token_valid, *, num_experts):
    """Per-shard per-expert sum of squared input activations plus assignment counts."""
    valid_f = token_valid.astype(jnp.float32)[:, None]
    squared = jnp.square(x.astype(jnp.float32)) * valid_f
    one_hot = jax.nn.one_hot(selected_experts, num_experts).astype(jnp.float32) * valid_f[:, None, :]
    sums = jnp.einsum("tke,td->ed", one_hot, squared)
    counts = jnp.sum(one_hot, axis=(0, 1))
    channel_sums = jnp.sum(squared, axis=0)
    channel_counts = jnp.sum(valid_f)
    return sums[None, :, :], counts[None, :], channel_sums[None, :], channel_counts[None]


def wanda_input_sums_local(
    x: jax.Array,
    selected_experts: jax.Array,
    token_valid: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    *,
    batch_axes: tuple[str, ...],
    num_experts: int,
) -> dict[str, jax.Array]:
    """Shard-local Wanda calibration statistics, stacked over layers by the caller."""
    sums, counts, channel_sums, channel_counts = shard_map(
        functools.partial(_local_wanda_sums, num_experts=num_experts),
        mesh=mesh,
        in_specs=(P(batch_axes, None), P(batch_axes, None), P(batch_axes)),
        out_specs=(P(batch_axes, None, None), P(batch_axes, None), P(batch_axes, None), P(batch_axes)),
    )(x, selected_experts, token_valid)
    return {
        "wanda_input_sums_local": sums,
        "wanda_input_counts_local": counts,
        "wanda_channel_sums_local": channel_sums,
        "wanda_channel_counts_local": channel_counts,
    }


def _expert_input_norms(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-expert mean-square norms with a layer-mean fallback for unrouted experts."""
    routed = counts > 0
    safe_counts = np.maximum(counts, 1.0)
    per_expert = np.divide(sums, safe_counts[..., None], out=np.zeros_like(sums), where=routed[..., None])
    routed_mean = per_expert[routed].mean(axis=0) if routed.any() else np.zeros(sums.shape[-1], dtype=sums.dtype)
    filled = np.where(routed[..., None], per_expert, routed_mean[None, :])
    return np.sqrt(np.maximum(filled, 0.0))


def wanda_delete_threshold(samples: np.ndarray, eta: float) -> float:
    """The delete-group threshold: the ``floor(eta * n)``-th smallest pooled score."""
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must be in (0, 1), got {eta}")
    ordered = np.sort(samples, kind="stable")
    delete_count = math.floor(eta * ordered.size)
    if delete_count == 0:
        raise ValueError(f"subsample too small for eta={eta}: only {ordered.size} samples")
    return float(ordered[delete_count - 1])


def down_norms(norms: jax.Array) -> jax.Array:
    """Collapse per-input-channel norms to one scalar per expert (recorded V3 deviation)."""
    return jnp.mean(norms, axis=-1)


def init_gate_logits_jax(scores: jax.Array, eta: float, key: PRNGKeyArray, threshold: float) -> jax.Array:
    """Per-matrix noisy gate logits under the threshold rule (paper Eq. 8 init)."""
    keep_key, delete_key = jax.random.split(key)
    # The noise draws from a replicated key; the select needs the cases sharded
    # like the scores under an explicit mesh.
    spec = jax.typeof(scores).sharding.spec
    keep_noise = jax.sharding.reshard(
        jax.random.normal(keep_key, scores.shape) * DeletionTuningDefaults["logit_std"]
        + DeletionTuningDefaults["keep_logit_mean"],
        spec,
    )
    delete_noise = jax.sharding.reshard(
        jax.random.normal(delete_key, scores.shape) * DeletionTuningDefaults["logit_std"]
        + DeletionTuningDefaults["delete_logit_mean"],
        spec,
    )
    return jnp.where(scores <= threshold, delete_noise, keep_noise).astype(scores.dtype)


def alpha_ramp(step: int, total_steps: int, alpha_start: float, alpha_end: float) -> float:
    """Linear mask-sharpening ramp, clamped to the endpoints."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    fraction = min(max(step / total_steps, 0.0), 1.0)
    return alpha_start + fraction * (alpha_end - alpha_start)


def deletion_regularizer(transformer, lam: float) -> jax.Array:
    """The sparse-weight term ``-lambda * sum(||W~||^2)`` over the masked weights.

    The pretrained weights enter through ``stop_gradient`` (inside
    :func:`deletion_mask`), so the term's gradient reaches only the gate
    logits. Used as the model's aux loss during training.
    """
    alpha = transformer.deletion_alpha
    hard = transformer.deletion_hard
    if alpha is None or hard is None:
        raise ValueError("deletion_regularizer requires a deletion-tuning model")
    total = jnp.zeros((), dtype=jnp.float32)

    def _masked_square_sum(weight: jax.Array, logits: jax.Array) -> jax.Array:
        return jnp.sum(jnp.square(deletion_mask(logits, weight, alpha, hard).astype(jnp.float32)))

    for block in transformer.blocks:
        mlp = block.mlp
        total = total + _masked_square_sum(mlp.expert_mlp.w_gate, mlp.gate_logits)
        if mlp.up_logits is not None:
            total = total + _masked_square_sum(mlp.expert_mlp.w_up, mlp.up_logits)
        if mlp.down_logits is not None:
            total = total + _masked_square_sum(mlp.expert_mlp.w_down, mlp.down_logits)
        if block.shared.gate_logits is not None:
            total = total + _masked_square_sum(block.shared.w_gate, block.shared.gate_logits)
            if block.shared.up_logits is not None:
                total = total + _masked_square_sum(block.shared.w_up, block.shared.up_logits)
            if block.shared.down_logits is not None:
                total = total + _masked_square_sum(block.shared.w_down, block.shared.down_logits)
    return -lam * total


def pack_keep_masks(logits: jax.Array) -> jax.Array:
    """Bit-pack a stacked keep mask on device (``np.packbits`` bit order)."""
    bits = logits > 0
    lead = bits.shape[:-1]
    flat = bits.reshape((*lead, -1))
    pad = (-flat.shape[-1]) % 8
    if pad:
        flat = jnp.pad(flat, [(0, 0)] * len(lead) + [(0, pad)])
    groups = flat.reshape((*lead, flat.shape[-1] // 8, 8)).astype(jnp.uint8)
    weights = jnp.asarray([128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.uint8)
    return jnp.sum(groups * weights, axis=-1).astype(jnp.uint8)


def _masked_matrices(model):
    """Yield ``(path, weight, logits)`` for every masked matrix carrying logits."""
    for index, block in enumerate(model.transformer.blocks):
        mlp = block.mlp
        if mlp.gate_logits is not None:
            yield f"transformer.blocks.{index}.mlp.expert_mlp.w_gate", mlp.expert_mlp.w_gate, mlp.gate_logits
        if mlp.up_logits is not None:
            yield f"transformer.blocks.{index}.mlp.expert_mlp.w_up", mlp.expert_mlp.w_up, mlp.up_logits
        if mlp.down_logits is not None:
            yield f"transformer.blocks.{index}.mlp.expert_mlp.w_down", mlp.expert_mlp.w_down, mlp.down_logits
        shared = block.shared
        if shared.gate_logits is not None:
            yield f"transformer.blocks.{index}.shared.w_gate", shared.w_gate, shared.gate_logits
        if shared.up_logits is not None:
            yield f"transformer.blocks.{index}.shared.w_up", shared.w_up, shared.up_logits
        if shared.down_logits is not None:
            yield f"transformer.blocks.{index}.shared.w_down", shared.w_down, shared.down_logits


def deletion_keep_masks(model) -> dict[str, np.ndarray]:
    """Boolean keep masks per masked matrix, keyed by the Levanter parameter path."""
    masks: dict[str, np.ndarray] = {}
    for path, _weight, logits in _masked_matrices(model):
        masks[path] = np.asarray(logits > 0)
    return masks


def apply_hard_mask(model):
    """The deployment model: deleted weights zeroed and the gate logits dropped.

    The exported checkpoint carries plain weights (deleted entries zero), so a
    standard reload reproduces the hard-mask forward exactly.
    """
    updated = model
    for path, weight, logits in _masked_matrices(model):
        parts = tuple(path.split("."))
        weight_field = parts[-1]
        logits_field = weight_field.replace("w_", "", 1) + "_logits"
        # routed logits live on the MoE wrapper (one hop above expert_mlp); shared logits on the DenseMLP
        logits_parts = parts[:-2] + (logits_field,) if "expert_mlp" in parts else parts[:-1] + (logits_field,)

        def navigate(obj, parts=parts):
            for part in parts:
                obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
            return obj

        def weight_where(m, parts=parts):
            return navigate(m, parts)

        def logits_where(m, logits_parts=logits_parts):
            return navigate(m, logits_parts)

        keep = (logits > 0).astype(weight.dtype)
        updated = _tree_replace(weight_where, updated, weight * keep)
        updated = _tree_replace(logits_where, updated, None)
    # the deployed checkpoint is a plain Snowball: no deletion state at all
    updated = _tree_replace(lambda m: m.transformer.deletion_alpha, updated, None)
    updated = _tree_replace(lambda m: m.transformer.deletion_hard, updated, None)
    return updated
