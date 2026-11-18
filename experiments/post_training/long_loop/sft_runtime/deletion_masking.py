# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deletion Tuning for the Snowball SFT runtime (arXiv 2611.10866).

The runtime trains one logit per masked expert weight and never moves the
pretrained weights: the forward reads ``W * sigmoid(alpha * G)`` (soft) during
training and ``W * 1[G > 0]`` (hard) for evaluation and export, the objective
adds ``-lambda * ||W~||^2``, and gate logits start from Wanda scores captured
on a calibration pass over the run's own training data. Only the gate logits
receive optimizer state; the frozen weights cannot move because their
gradients are partitioned out of the update.

Scope meanings:
- ``routed_gate`` masks the routed experts' ``gate_proj`` (the paper's scope).
- ``routed_and_shared_all`` additionally masks routed and shared experts'
  ``up_proj`` and ``down_proj`` (the wider V3 scope). For the wider scope the
  Wanda scores reuse the expert-input norms the gate scope captures; the
  down projection's true input lives in the intermediate width, and capturing
  it would need a second hook inside the expert MLP.
"""

import dataclasses
import math
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray
from levanter.data.text.examples import GrugLmExample
from marin.rl.deletion_tuning import (
    DeletionTuningDefaults,
    MaskScopeStats,
    build_mask_artifact,
    mask_scope_stats,
)

from experiments.post_training.long_loop.sft_runtime.model import (
    Transformer,
    deletion_mask,
)

_LOGIT_FIELD_NAMES = frozenset({"gate_logits", "up_logits", "down_logits"})

WANDA_SAMPLE_STRIDE = 64
"""Output-axis stride for the Wanda score subsample that fixes the delete threshold."""


class DeletionTuningScope(StrEnum):
    ROUTED_GATE = "routed_gate"
    ROUTED_AND_SHARED_ALL = "routed_and_shared_all"


@dataclass
class DeletionTuningSettings:
    """Deletion Tuning run settings; the paper's defaults unless stated."""

    scope: DeletionTuningScope = DeletionTuningScope.ROUTED_GATE
    eta: float = DeletionTuningDefaults.eta
    lam: float = DeletionTuningDefaults.lam
    alpha_start: float = DeletionTuningDefaults.alpha_start_sft
    alpha_end: float = DeletionTuningDefaults.alpha_end_sft
    lr: float = 5e-4
    """Adam learning rate for the gate logits (paper SFT: 5e-4, GRPO: 3e-4)."""
    calibration_sequences: int = DeletionTuningDefaults.wanda_calibration_prompts
    seed: int = 0
    total_steps: int | None = None
    """Training steps the alpha ramp spans; the launcher fills this from the trainer config."""

    def __post_init__(self) -> None:
        if not 0.0 < self.eta < 1.0:
            raise ValueError(f"eta must be in (0, 1), got {self.eta}")
        if self.lam < 0:
            raise ValueError(f"lam must be non-negative, got {self.lam}")
        if self.alpha_start <= 0 or self.alpha_end <= self.alpha_start:
            raise ValueError(f"alpha must ramp upward: start={self.alpha_start}, end={self.alpha_end}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.total_steps is not None and self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")

    @property
    def masks_up_and_down(self) -> bool:
        return self.scope == DeletionTuningScope.ROUTED_AND_SHARED_ALL

    @property
    def masks_shared_experts(self) -> bool:
        return self.scope == DeletionTuningScope.ROUTED_AND_SHARED_ALL


@dataclass(frozen=True)
class DeletionTrainConfig:
    """Static per-step deletion-tuning inputs captured by the train-step closure."""

    settings: DeletionTuningSettings
    filter_tree: object
    """eqx filter tree marking the trainable gate-logit leaves."""
    total_steps: int

    def __post_init__(self) -> None:
        if self.settings.total_steps is None:
            object.__setattr__(self.settings, "total_steps", self.total_steps)
        elif self.settings.total_steps != self.total_steps:
            raise ValueError(
                f"settings.total_steps={self.settings.total_steps} disagrees with the trainer's {self.total_steps}"
            )


def alpha_ramp_jax(step: jax.Array, total_steps: int, alpha_start: float, alpha_end: float) -> jax.Array:
    """The linear alpha ramp as a traced scalar, clamped to the endpoints."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    fraction = jnp.clip(step.astype(jnp.float32) / jnp.asarray(total_steps, jnp.float32), 0.0, 1.0)
    return alpha_start + fraction * (alpha_end - alpha_start)


def deletion_regularizer(model: Transformer, lam: float) -> jax.Array:
    """The sparse-weight term ``-lambda * sum(||W~||^2)`` over the masked weights.

    The pretrained weights enter through ``stop_gradient``, so the term's
    gradient reaches only the gate logits.
    """
    alpha = model.deletion_alpha
    hard = model.deletion_hard
    if alpha is None or hard is None:
        raise ValueError("deletion_regularizer requires a deletion-tuning model")
    blocks = model.stacked_blocks.stacked
    total = jnp.zeros((), dtype=jnp.float32)

    def _masked_square_sum(weight: jax.Array, logits: jax.Array) -> jax.Array:
        return jnp.sum(jnp.square(deletion_mask(logits, weight, alpha, hard).astype(jnp.float32)))

    mlp = blocks.mlp
    total = total + _masked_square_sum(mlp.expert_mlp.w_gate, mlp.gate_logits)
    if mlp.up_logits is not None:
        total = total + _masked_square_sum(mlp.expert_mlp.w_up, mlp.up_logits)
    if mlp.down_logits is not None:
        total = total + _masked_square_sum(mlp.expert_mlp.w_down, mlp.down_logits)
    if blocks.shared is not None:
        for shared in blocks.shared:
            if shared.gate_logits is not None:
                total = total + _masked_square_sum(shared.w_gate, shared.gate_logits)
            if shared.up_logits is not None:
                total = total + _masked_square_sum(shared.w_up, shared.up_logits)
            if shared.down_logits is not None:
                total = total + _masked_square_sum(shared.w_down, shared.down_logits)
    return -lam * total


def deletion_sparsity(model: Transformer) -> jax.Array:
    """The fraction of masked weights the current hard mask deletes."""
    blocks = model.stacked_blocks.stacked
    deleted = jnp.zeros((), dtype=jnp.float32)
    total = jnp.zeros((), dtype=jnp.float32)

    def _count(logits: jax.Array | None) -> tuple[jax.Array, jax.Array] | None:
        if logits is None:
            return None
        return jnp.sum((logits <= 0).astype(jnp.float32)), jnp.asarray(logits.size, jnp.float32)

    mlp = blocks.mlp
    for logits in (mlp.gate_logits, mlp.up_logits, mlp.down_logits):
        counts = _count(logits)
        if counts is not None:
            deleted = deleted + counts[0]
            total = total + counts[1]
    if blocks.shared is not None:
        for shared in blocks.shared:
            for logits in (shared.gate_logits, shared.up_logits, shared.down_logits):
                counts = _count(logits)
                if counts is not None:
                    deleted = deleted + counts[0]
                    total = total + counts[1]
    return deleted / total


def deletion_logit_filter(model: Transformer):
    """A filter tree marking the gate-logit leaves (True) and everything else (False).

    Pass it to ``eqx.partition`` to split parameters or gradients into the
    trainable logits and the frozen weights.
    """
    leaves, treedef = jax.tree_util.tree_flatten_with_path(model)
    keep = []
    for path, _leaf in leaves:
        key = getattr(path[-1], "name", None) or getattr(path[-1], "key", None)
        keep.append(key in _LOGIT_FIELD_NAMES)
    return treedef.unflatten(keep)


def _expert_input_norms(sums: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-expert mean-square norms with a layer-mean fallback for unrouted experts."""
    routed = counts > 0
    safe_counts = np.maximum(counts, 1.0)
    per_expert = np.divide(sums, safe_counts[..., None], out=np.zeros_like(sums), where=routed[..., None])
    if routed.any():
        routed_mean = per_expert[routed].mean(axis=0)
    else:
        routed_mean = np.zeros(sums.shape[-1], dtype=sums.dtype)
    filled = np.where(routed[..., None], per_expert, routed_mean[None, :])
    return np.sqrt(np.maximum(filled, 0.0)), routed_mean


def capture_expert_input_norms(
    model: Transformer,
    batches: Iterator[GrugLmExample],
    *,
    max_sequences: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the base model over calibration batches and return Wanda input norms.

    Returns:
        ``(routed_norms, shared_norms)`` where ``routed_norms`` is
        ``[num_layers, num_experts, expert_input_width]`` (experts with no
        routed assignment fall back to their layer's routed mean) and
        ``shared_norms`` is ``[num_layers, expert_input_width]``, the norm of
        the block MLP input every shared expert reads.
    """
    if model.stacked_blocks.stacked.mlp.gate_logits is not None:
        raise ValueError("capture must run on the base model before deletion init")
    stacked = model.stacked_blocks.stacked
    capture_stacked = eqx.tree_at(lambda s: s.mlp, stacked, dataclasses.replace(stacked.mlp, capture_input_sums=True))
    capture_model = eqx.tree_at(lambda m: m.stacked_blocks.stacked, model, capture_stacked)

    @jax.jit
    def capture_step(batch: GrugLmExample):
        _, metrics = capture_model(batch.tokens, mask=batch.attn_mask)
        return (
            metrics["wanda_input_sums_per_layer"],
            metrics["wanda_input_counts_per_layer"],
            metrics["wanda_channel_sums_per_layer"],
            metrics["wanda_channel_counts_per_layer"],
        )

    sums_total: np.ndarray | None = None
    counts_total: np.ndarray | None = None
    channel_sums_total: np.ndarray | None = None
    channel_counts_total: np.ndarray | None = None
    sequences_seen = 0
    for batch in batches:
        rows = batch.tokens.shape[0]
        if sequences_seen + rows > max_sequences:
            break
        sums, counts, channel_sums, channel_counts = capture_step(batch)
        sums_host = np.asarray(sums, dtype=np.float64)
        counts_host = np.asarray(counts, dtype=np.float64)
        channel_sums_host = np.asarray(channel_sums, dtype=np.float64)
        channel_counts_host = np.asarray(channel_counts, dtype=np.float64)
        sums_total = sums_host if sums_total is None else sums_total + sums_host
        counts_total = counts_host if counts_total is None else counts_total + counts_host
        channel_sums_total = channel_sums_host if channel_sums_total is None else channel_sums_total + channel_sums_host
        channel_counts_total = (
            channel_counts_host if channel_counts_total is None else channel_counts_total + channel_counts_host
        )
        sequences_seen += rows
    if sums_total is None or counts_total is None or channel_sums_total is None or channel_counts_total is None:
        raise ValueError("calibration produced no batches; cannot initialize Wanda norms")

    routed_norms, _ = _expert_input_norms(sums_total, counts_total)
    shared_norms = np.sqrt(np.maximum(channel_sums_total / np.maximum(channel_counts_total, 1.0)[:, None], 0.0))
    return routed_norms, shared_norms


def wanda_score_samples(
    model: Transformer, settings: DeletionTuningSettings, routed_norms, shared_norms, *, stride: int
) -> np.ndarray:
    """Strided subsample of the Wanda scores ``|W| * norm`` over the whole masked scope.

    The scores themselves are elementwise (sharding-safe); only the global
    eta-quantile needs cross-shard data, and the subsample carries it to the
    host instead of sorting a sharded array. Stride applies to each weight's
    output axis.
    """
    mlp = model.stacked_blocks.stacked.mlp
    norms_jax = jnp.asarray(routed_norms, dtype=jnp.float32)
    shared_norms_jax = jnp.asarray(shared_norms, dtype=jnp.float32)

    def broadcast(weight: jax.Array, norms: jax.Array) -> jax.Array:
        return norms.reshape(norms.shape + (1,) * (weight.ndim - norms.ndim))

    def per_scope(weight: jax.Array, norms: jax.Array) -> jax.Array:
        scores = jnp.abs(weight) * broadcast(weight, norms)
        strided = scores[..., ::stride]
        return strided.reshape(strided.shape[0], -1)

    @jax.jit
    def sample():
        parts = [per_scope(mlp.expert_mlp.w_gate, norms_jax)]
        if settings.masks_up_and_down:
            parts.append(per_scope(mlp.expert_mlp.w_up, norms_jax))
            parts.append(per_scope(mlp.expert_mlp.w_down, _down_norms(norms_jax)))
        if model.stacked_blocks.stacked.shared is not None:
            for shared in model.stacked_blocks.stacked.shared:
                parts.append(per_scope(shared.w_gate, shared_norms_jax))
                if settings.masks_shared_experts:
                    parts.append(per_scope(shared.w_up, shared_norms_jax))
                    parts.append(per_scope(shared.w_down, _down_norms(shared_norms_jax)))
        return jnp.concatenate(parts, axis=1)  # [L, samples per layer]

    stacked = np.asarray(sample())
    return stacked.reshape(-1)


def wanda_delete_threshold(samples: np.ndarray, eta: float) -> float:
    """The delete-group threshold: the ``floor(eta * n)``-th smallest score.

    Applies the core module's group rule (the lowest ``floor(eta * n)`` scores
    form the delete group) to the pooled subsample; the realized delete
    fraction on the full scope then matches eta up to subsampling noise.
    """
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must be in (0, 1), got {eta}")
    ordered = np.sort(samples, kind="stable")
    delete_count = math.floor(eta * ordered.size)
    if delete_count == 0:
        raise ValueError(f"subsample too small for eta={eta}: only {ordered.size} samples")
    return float(ordered[delete_count - 1])


def _tree_replace(where, model: Transformer, value):
    """``eqx.tree_at`` that can replace ``None`` leaves (the not-yet-initialized logits)."""
    return eqx.tree_at(where, model, value, is_leaf=lambda x: x is None)


def _reshard_like(logits: jax.Array, reference: jax.Array) -> jax.Array:
    return jax.sharding.reshard(logits, jax.typeof(reference).sharding.spec)


def _layer_logits(
    weight: jax.Array,
    norms: jax.Array,
    keys: jax.Array,
    settings: DeletionTuningSettings,
    threshold: float,
) -> jax.Array:
    """Per-layer gate logits ``[L, ...]`` from a stacked weight, its input norms, and the threshold.

    ``weight`` is ``[L, ...]`` with the input-channel axis at position -2 (the
    stacked ``[..., d_in, d_out]`` layout). ``norms`` supplies one value per
    input channel when its last axis matches ``d_in``; a lower-rank norms array
    (for example one scalar per expert for the down projection, whose true
    per-channel norms are not captured) broadcasts as-is.
    """

    def per_layer(w_layer, norms_layer, key_layer):
        broadcast = norms_layer.reshape(norms_layer.shape + (1,) * (w_layer.ndim - norms_layer.ndim))
        scores = jnp.abs(w_layer) * broadcast
        keep_key, delete_key = jax.random.split(key_layer)
        score_spec = jax.typeof(scores).sharding.spec
        keep_noise = jax.sharding.reshard(
            jax.random.normal(keep_key, scores.shape) * DeletionTuningDefaults.logit_std
            + DeletionTuningDefaults.keep_logit_mean,
            score_spec,
        )
        delete_noise = jax.sharding.reshard(
            jax.random.normal(delete_key, scores.shape) * DeletionTuningDefaults.logit_std
            + DeletionTuningDefaults.delete_logit_mean,
            score_spec,
        )
        logits = jnp.where(scores <= threshold, delete_noise, keep_noise)
        return logits.astype(w_layer.dtype)

    return jax.vmap(per_layer)(weight, norms, keys)


def _down_norms(norms: jax.Array) -> jax.Array:
    """Collapse per-input-channel norms to one scalar per expert.

    The down projection's input lives in the intermediate width, which the
    calibration pass does not capture; its Wanda scores reuse the expert's
    mean input norm as a per-expert scale (recorded deviation).
    """
    return jnp.mean(norms, axis=-1)


def apply_deletion_init(
    model: Transformer,
    settings: DeletionTuningSettings,
    routed_norms: np.ndarray,
    shared_norms: np.ndarray,
    *,
    threshold: float,
    key: PRNGKeyArray,
) -> Transformer:
    """Insert stacked gate logits initialized from Wanda scores and set the ramp start.

    ``routed_norms`` is ``[L, E, D]`` and ``shared_norms`` is ``[L, D]`` from
    :func:`capture_expert_input_norms`; ``threshold`` comes from
    :func:`wanda_delete_threshold` over the whole masked scope. Logit noise is
    rank-local and deterministic given the seed, threshold, and mesh.
    """
    mlp = model.stacked_blocks.stacked.mlp
    num_layers = model.config.num_layers
    layer_keys = jax.random.split(key, num_layers)
    norms_jax = jnp.asarray(routed_norms, dtype=jnp.float32)
    shared_norms_jax = jnp.asarray(shared_norms, dtype=jnp.float32)

    updated = model
    updated = _tree_replace(
        lambda m: m.stacked_blocks.stacked.mlp.gate_logits,
        updated,
        _reshard_like(
            _layer_logits(mlp.expert_mlp.w_gate, norms_jax, layer_keys, settings, threshold),
            mlp.expert_mlp.w_gate,
        ),
    )
    if settings.masks_up_and_down:
        updated = _tree_replace(
            lambda m: m.stacked_blocks.stacked.mlp.up_logits,
            updated,
            _reshard_like(
                _layer_logits(mlp.expert_mlp.w_up, norms_jax, layer_keys, settings, threshold),
                mlp.expert_mlp.w_up,
            ),
        )
        updated = _tree_replace(
            lambda m: m.stacked_blocks.stacked.mlp.down_logits,
            updated,
            _reshard_like(
                _layer_logits(mlp.expert_mlp.w_down, _down_norms(norms_jax), layer_keys, settings, threshold),
                mlp.expert_mlp.w_down,
            ),
        )
    shared_stack = model.stacked_blocks.stacked.shared
    if shared_stack is not None:
        for shared_index, shared in enumerate(shared_stack):
            updated = _tree_replace(
                lambda m, index=shared_index: m.stacked_blocks.stacked.shared[index].gate_logits,
                updated,
                _reshard_like(
                    _layer_logits(shared.w_gate, shared_norms_jax, layer_keys, settings, threshold),
                    shared.w_gate,
                ),
            )
            if settings.masks_shared_experts:
                updated = _tree_replace(
                    lambda m, index=shared_index: m.stacked_blocks.stacked.shared[index].up_logits,
                    updated,
                    _reshard_like(
                        _layer_logits(shared.w_up, shared_norms_jax, layer_keys, settings, threshold),
                        shared.w_up,
                    ),
                )
                updated = _tree_replace(
                    lambda m, index=shared_index: m.stacked_blocks.stacked.shared[index].down_logits,
                    updated,
                    _reshard_like(
                        _layer_logits(shared.w_down, _down_norms(shared_norms_jax), layer_keys, settings, threshold),
                        shared.w_down,
                    ),
                )

    updated = _tree_replace(
        lambda m: m.deletion_alpha,
        updated,
        jnp.asarray(settings.alpha_start, dtype=jnp.float32),
    )
    # 0.0 applies the soft mask and 1.0 the hard one; a float leaf keeps the
    # train step's value_and_grad happy (jax.grad rejects bool inputs).
    updated = _tree_replace(lambda m: m.deletion_hard, updated, jnp.asarray(0.0))
    return updated


def with_deletion_eval_mode(model: Transformer) -> Transformer:
    """A copy of the model whose forwards use the hard mask (deploy semantics)."""
    if model.deletion_hard is None:
        raise ValueError("with_deletion_eval_mode requires a deletion-tuning model")
    return _tree_replace(lambda m: m.deletion_hard, model, jnp.asarray(1.0))


def apply_hard_mask(model: Transformer) -> Transformer:
    """Zero every deleted weight in place, for export; the logits stay intact."""
    blocks = model.stacked_blocks.stacked
    alpha = model.deletion_alpha
    hard = model.deletion_hard
    if alpha is None or hard is None:
        raise ValueError("apply_hard_mask requires a deletion-tuning model")
    true_hard = jnp.asarray(1.0)

    def _masked(weight: jax.Array, logits: jax.Array) -> jax.Array:
        return deletion_mask(logits, weight, alpha, true_hard)

    mlp = blocks.mlp
    updated = model
    updated = _tree_replace(
        lambda m: m.stacked_blocks.stacked.mlp.expert_mlp.w_gate,
        updated,
        _masked(mlp.expert_mlp.w_gate, mlp.gate_logits),
    )
    if mlp.up_logits is not None:
        updated = _tree_replace(
            lambda m: m.stacked_blocks.stacked.mlp.expert_mlp.w_up,
            updated,
            _masked(mlp.expert_mlp.w_up, mlp.up_logits),
        )
    if mlp.down_logits is not None:
        updated = _tree_replace(
            lambda m: m.stacked_blocks.stacked.mlp.expert_mlp.w_down,
            updated,
            _masked(mlp.expert_mlp.w_down, mlp.down_logits),
        )
    if blocks.shared is not None:
        for shared_index, shared in enumerate(blocks.shared):
            if shared.gate_logits is not None:
                updated = _tree_replace(
                    lambda m, index=shared_index: m.stacked_blocks.stacked.shared[index].w_gate,
                    updated,
                    _masked(shared.w_gate, shared.gate_logits),
                )
            if shared.up_logits is not None:
                updated = _tree_replace(
                    lambda m, index=shared_index: m.stacked_blocks.stacked.shared[index].w_up,
                    updated,
                    _masked(shared.w_up, shared.up_logits),
                )
            if shared.down_logits is not None:
                updated = _tree_replace(
                    lambda m, index=shared_index: m.stacked_blocks.stacked.shared[index].w_down,
                    updated,
                    _masked(shared.w_down, shared.down_logits),
                )
    return updated


def _device_keep_masks(logits: jax.Array) -> np.ndarray:
    """Host-side keep mask for one stacked logit array ``[L, ...]``, one layer at a time."""
    layers = []
    for layer_index in range(logits.shape[0]):
        layers.append(np.asarray(logits[layer_index] > 0))
    return np.stack(layers)


def pack_keep_masks_device(logits: jax.Array) -> jax.Array:
    """Bit-pack a stacked keep mask on device: ``[L, ..., n]`` bool -> ``[L, ..., n/8]`` uint8.

    Bit order matches ``np.packbits`` (big-endian within each byte), so the
    core artifact reader unpacks these bytes unchanged. The last axis is
    zero-padded to a multiple of 8; the manifest records the logical shape.
    """
    bits = logits > 0
    lead = bits.shape[:-1]
    flat = bits.reshape((*lead, -1))
    pad = (-flat.shape[-1]) % 8
    if pad:
        flat = jnp.pad(flat, [(0, 0)] * len(lead) + [(0, pad)])
    groups = flat.reshape((*lead, flat.shape[-1] // 8, 8)).astype(jnp.uint8)
    weights = jnp.asarray([128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.uint8)
    return jnp.sum(groups * weights, axis=-1).astype(jnp.uint8)


def deletion_mask_state_dict(model: Transformer) -> dict[str, jax.Array]:
    """Packed keep-mask tensors keyed by the exported tensor names.

    One packed uint8 tensor per stacked (bank, projection) group, named after
    the exported stacked weight with a ``deletion_mask/`` prefix, plus an int32
    shape tensor recording the exported (transposed) per-expert weight shape.
    Bit order matches ``np.packbits``.
    """
    if model.deletion_alpha is None:
        raise ValueError("deletion_mask_state_dict requires a deletion-tuning model")
    blocks = model.stacked_blocks.stacked
    tensors: dict[str, jax.Array] = {}
    mlp = blocks.mlp
    scoped = (
        ("experts", "gate_proj", mlp.gate_logits, mlp.expert_mlp.w_gate),
        ("experts", "up_proj", mlp.up_logits, mlp.expert_mlp.w_up),
        ("experts", "down_proj", mlp.down_logits, mlp.expert_mlp.w_down),
    )
    for bank, projection, logits, weight in scoped:
        if logits is None:
            continue
        packed = pack_keep_masks_device(logits)
        layer, experts, in_dim, out_dim = weight.shape
        exported_shape = (layer, experts, out_dim, in_dim)
        tensors[f"deletion_mask/{bank}.{projection}.weight"] = packed
        tensors[f"deletion_mask/shapes/{bank}.{projection}.weight"] = jnp.asarray(exported_shape, dtype=jnp.int32)
    if blocks.shared is not None:
        for shared_index, shared in enumerate(blocks.shared):
            for projection, logits, weight in (
                ("gate_proj", shared.gate_logits, shared.w_gate),
                ("up_proj", shared.up_logits, shared.w_up),
                ("down_proj", shared.down_logits, shared.w_down),
            ):
                if logits is None:
                    continue
                packed = pack_keep_masks_device(logits)
                layer, in_dim, out_dim = weight.shape
                exported_shape = (layer, out_dim, in_dim)
                tensors[f"deletion_mask/shared_experts.{shared_index}.{projection}.weight"] = packed
                tensors[f"deletion_mask/shapes/shared_experts.{shared_index}.{projection}.weight"] = jnp.asarray(
                    exported_shape, dtype=jnp.int32
                )
    return tensors


def hard_keep_tensors(model: Transformer, num_experts: int) -> dict[str, np.ndarray]:
    """Keep masks keyed by the exported per-expert HF tensor names.

    The exported layout transposes the JAX weight (``_linear_inference_tensor``
    swaps the last two axes), so each mask is transposed the same way.
    """
    if model.deletion_hard is None:
        raise ValueError("hard_keep_tensors requires a deletion-tuning model")
    blocks = model.stacked_blocks.stacked
    masks: dict[str, np.ndarray] = {}
    mlp = blocks.mlp
    scoped = (
        ("gate_proj", mlp.gate_logits),
        ("up_proj", mlp.up_logits),
        ("down_proj", mlp.down_logits),
    )
    for projection, logits in scoped:
        if logits is None:
            continue
        keep = _device_keep_masks(logits)  # [L, ...weight axes...]
        for layer_index in range(keep.shape[0]):
            for expert_index in range(num_experts):
                name = f"model.layers.{layer_index}.mlp.experts.{expert_index}.{projection}.weight"
                masks[name] = np.transpose(keep[layer_index][expert_index], (1, 0))
    if blocks.shared is not None:
        for shared_index, shared in enumerate(blocks.shared):
            for projection, logits in (
                ("gate_proj", shared.gate_logits),
                ("up_proj", shared.up_logits),
                ("down_proj", shared.down_logits),
            ):
                if logits is None:
                    continue
                keep = _device_keep_masks(logits)
                for layer_index in range(keep.shape[0]):
                    name = f"model.layers.{layer_index}.shared_experts.{shared_index}.{projection}.weight"
                    masks[name] = np.transpose(keep[layer_index], (1, 0))
    return masks


def deletion_mask_stats(model: Transformer, *, total_model_weights: int) -> MaskScopeStats:
    """Learned sparsity and compressed mask size for the trained model."""
    keep = hard_keep_tensors(model, model.config.num_experts)
    return mask_scope_stats(keep, total_model_weights=total_model_weights)


def build_export_artifacts(model: Transformer) -> tuple[dict, dict[str, bytes]]:
    """Manifest and zstd blobs for the trained mask, keyed by exported tensor names."""
    keep = hard_keep_tensors(model, model.config.num_experts)
    return build_mask_artifact(keep)
