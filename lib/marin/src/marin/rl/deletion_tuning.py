# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deletion Tuning: post-training by learning which pretrained weights to zero.

Implements the masking core of Deletion Tuning (arXiv 2611.10866). A trainable
logit ``G`` (same shape as the masked weight ``W``) decides whether each weight
is kept or deleted. Training forwards use the soft mask ``W * sigmoid(alpha * G)``
with a linear ``alpha`` ramp; evaluation, rollouts, and export use the hard mask
``W * 1[G > 0]``. Gate logits start from Wanda scores (``|W| * ||x||`` over a
calibration sample), and the training objective adds the MaskLLM sparse-weight
term ``-lambda * ||W~||^2``, which mildly biases the mask toward retaining
higher-magnitude weights. The pretrained weights stay frozen; ``G`` is the only
trainable parameter in the masked scope.

This module holds the framework-neutral pieces: the ramp, Wanda scoring, gate
logit initialization, the sparse-weight term, hard-mask export, and the compact
bit-packed + zstd mask artifact. Training runtimes apply the same mask math
in their own framework (one sigmoid and one multiply) and freeze every weight
outside ``G``.

The paper's defaults live in :class:`DeletionTuningDefaults`. Deviating from
them requires a recorded reason in the experiment's decision log.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zstandard

MASK_ARTIFACT_FORMAT = "marin-deletion-tuning-mask-v1"
MASK_MANIFEST_NAME = "manifest.json"

_KEEP = True
"""Keep-mask convention: ``True`` retains the pretrained value, ``False`` zeroes it."""


@dataclasses.dataclass(frozen=True)
class DeletionTuningDefaults:
    """The paper's hyperparameters, shared by the SFT and GRPO regimes."""

    eta: float = 1 / 2048
    """Fraction of masked-scope weights initialized into the delete group."""
    keep_logit_mean: float = 0.03
    delete_logit_mean: float = -0.03
    logit_std: float = 0.01
    lam: float = 1e-5
    alpha_start_sft: float = 1e2
    alpha_end_sft: float = 1e3
    alpha_start_rl: float = 1e3
    alpha_end_rl: float = 1e4
    wanda_calibration_prompts: int = 2048


def alpha_ramp(step: int, total_steps: int, alpha_start: float, alpha_end: float) -> float:
    """Linear mask-sharpening ramp from ``alpha_start`` to ``alpha_end``.

    Args:
        step: Current training step in ``[0, total_steps]``; values outside the
            range clamp to the endpoints.
        total_steps: Total training steps ``T``.
        alpha_start: Sharpness at step 0.
        alpha_end: Sharpness at step ``T``.

    Returns:
        The ramp value ``alpha(step)``.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    fraction = min(max(step / total_steps, 0.0), 1.0)
    return alpha_start + fraction * (alpha_end - alpha_start)


def wanda_scores(weight: np.ndarray, input_norms: np.ndarray, *, input_axis: int = -1) -> np.ndarray:
    """Wanda importance scores ``s_ij = |W_ij| * ||X_j||_2``.

    Args:
        weight: The pretrained weight. The input-channel axis holds the columns
            the activation norms are indexed by: the last axis for a torch
            ``nn.Linear`` weight ``[d_out, d_in]``, axis -2 for the stacked
            JAX layout ``[..., d_in, d_out]``.
        input_norms: Per-input-channel activation norms from one forward pass
            over calibration prompts, shape ``[d_in]``.
        input_axis: Which axis of ``weight`` is the input-channel axis.

    Returns:
        Scores with the same shape as ``weight``.
    """
    d_in = weight.shape[input_axis]
    if input_norms.shape != (d_in,):
        raise ValueError(f"input_norms shape {input_norms.shape} must be [d_in={d_in}] for weight shape {weight.shape}")
    broadcast_shape = [1] * weight.ndim
    broadcast_shape[input_axis] = d_in
    return np.abs(weight) * input_norms.reshape(broadcast_shape)


def init_gate_logits(
    scores: np.ndarray,
    eta: float,
    seed: int,
    *,
    keep_logit_mean: float = DeletionTuningDefaults.keep_logit_mean,
    delete_logit_mean: float = DeletionTuningDefaults.delete_logit_mean,
    logit_std: float = DeletionTuningDefaults.logit_std,
) -> np.ndarray:
    """Sample per-weight gate logits from the Wanda prior (paper Eq. 8).

    The lowest-``eta`` fraction of weights by score enter the delete group and
    sample ``N(delete_logit_mean, logit_std^2)``; the rest enter the keep group
    and sample ``N(keep_logit_mean, logit_std^2)``. With the defaults, about
    ``eta * Phi(3) + (1 - eta) * Phi(-3)`` (0.18% at ``eta = 1/2048``) of the
    masked weights start deleted under the hard mask.

    Args:
        scores: Wanda scores from :func:`wanda_scores`.
        eta: Fraction of weights in the delete group; the count is
            ``floor(eta * n)``.
        seed: Seed for the logit noise; the group assignment is deterministic.
        keep_logit_mean: Mean logit for the keep group.
        delete_logit_mean: Mean logit for the delete group.
        logit_std: Standard deviation of both groups.

    Returns:
        Gate logits with the same shape as ``scores``.
    """
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must be in (0, 1), got {eta}")
    flat_scores = scores.reshape(-1)
    count = flat_scores.size
    delete_count = math.floor(eta * count)
    order = np.argsort(flat_scores, kind="stable")
    delete_group = np.zeros(count, dtype=bool)
    delete_group[order[:delete_count]] = True

    rng = np.random.default_rng(seed)
    logits = rng.normal(keep_logit_mean, logit_std, size=count)
    delete_logits = rng.normal(delete_logit_mean, logit_std, size=count)
    logits[delete_group] = delete_logits[delete_group]
    return logits.astype(np.float32).reshape(scores.shape)


def soft_mask(logits: np.ndarray, alpha: float) -> np.ndarray:
    """The training-time soft mask ``sigmoid(alpha * G)``."""
    return 1.0 / (1.0 + np.exp(-alpha * logits))


def hard_keep_mask(logits: np.ndarray) -> np.ndarray:
    """The deployment hard mask ``1[G > 0]`` as booleans (True keeps the weight)."""
    return logits > 0


def apply_keep_mask(weight: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Zero every deleted weight, leaving kept weights at their pretrained values."""
    if keep_mask.shape != weight.shape:
        raise ValueError(f"keep_mask shape {keep_mask.shape} does not match weight shape {weight.shape}")
    return np.where(keep_mask, weight, np.zeros((), dtype=weight.dtype)).astype(weight.dtype, copy=False)


def regularizer_loss(soft_masked_weights: Mapping[str, np.ndarray], lam: float) -> float:
    """The sparse-weight term ``-lambda * sum(||W~||^2)`` added to the task loss.

    The negative sign rewards large retained norms, biasing the mask toward
    keeping higher-magnitude weights (the paper's MaskLLM regularizer).
    """
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    total = 0.0
    for weight in soft_masked_weights.values():
        total += float(np.sum(np.square(weight, dtype=np.float64)))
    return -lam * total


@dataclass(frozen=True)
class MaskEntry:
    """One masked weight matrix inside a mask artifact."""

    name: str
    shape: tuple[int, ...]
    kept: int
    deleted: int
    sha256: str
    file_name: str


def build_mask_artifact(keep_masks: Mapping[str, np.ndarray]) -> tuple[dict, dict[str, bytes]]:
    """Build the mask artifact in memory: a manifest dict plus zstd blobs per entry.

    Use this where the artifact must travel through a storage layer that is not
    a local filesystem; :func:`write_mask_artifact` is the local-file wrapper.
    """
    entries = []
    blobs: dict[str, bytes] = {}
    for name, keep_mask in sorted(keep_masks.items()):
        if keep_mask.dtype != np.bool_:
            raise ValueError(f"keep mask for {name} must be boolean, got {keep_mask.dtype}")
        packed = np.packbits(keep_mask.reshape(-1)).tobytes()
        compressed = zstandard.compress(packed)
        file_name = f"{name}.mask.bin"
        blobs[file_name] = compressed
        entries.append(
            {
                "name": name,
                "shape": [int(dim) for dim in keep_mask.shape],
                "kept": int(np.count_nonzero(keep_mask)),
                "deleted": int(keep_mask.size - np.count_nonzero(keep_mask)),
                "sha256": hashlib.sha256(compressed).hexdigest(),
                "file_name": file_name,
            }
        )
    manifest = {"format": MASK_ARTIFACT_FORMAT, "entries": entries}
    return manifest, blobs


def write_mask_artifact(directory: Path | str, keep_masks: Mapping[str, np.ndarray]) -> list[MaskEntry]:
    """Write the compact adapter: bit-packed keep-masks, zstd-compressed.

    Each masked matrix becomes one zstd blob of ``np.packbits`` bytes of its
    keep-mask (row-major), plus a manifest entry with shape, counts, and a
    SHA-256 of the compressed bytes. Kept weights are read from the base
    checkpoint at deployment, so the artifact stores no floating-point values.

    Args:
        directory: Output directory; created if missing. Existing files with
            colliding names raise.
        keep_masks: Mapping of tensor name to boolean keep-mask.

    Returns:
        The written entries, in the manifest's order.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest, blobs = build_mask_artifact(keep_masks)
    entries: list[MaskEntry] = []
    for entry in manifest["entries"]:
        target = directory / entry["file_name"]
        if target.exists():
            raise FileExistsError(f"mask artifact file already exists: {target}")
        target.write_bytes(blobs[entry["file_name"]])
        entries.append(
            MaskEntry(
                name=entry["name"],
                shape=tuple(entry["shape"]),
                kept=entry["kept"],
                deleted=entry["deleted"],
                sha256=entry["sha256"],
                file_name=entry["file_name"],
            )
        )
    manifest_path = directory / MASK_MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(f"mask artifact manifest already exists: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return entries


def read_mask_artifact(directory: Path | str) -> dict[str, np.ndarray]:
    """Load a mask artifact back into boolean keep-masks, verifying digests.

    Raises:
        ValueError: If the manifest format is unknown or an entry's digest or
            shape does not match its file.
    """
    directory = Path(directory)
    manifest = json.loads((directory / MASK_MANIFEST_NAME).read_text())
    if manifest.get("format") != MASK_ARTIFACT_FORMAT:
        raise ValueError(f"unsupported mask artifact format: {manifest.get('format')!r}")
    keep_masks: dict[str, np.ndarray] = {}
    for entry in manifest["entries"]:
        blob = (directory / entry["file_name"]).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"mask artifact entry {entry['name']} failed its SHA-256 check")
        packed = zstandard.decompress(blob)
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
        expected_size = math.prod(entry["shape"])
        if bits.size < expected_size:
            raise ValueError(f"mask artifact entry {entry['name']} is truncated: {bits.size} < {expected_size} bits")
        keep_masks[entry["name"]] = bits[:expected_size].astype(bool).reshape(entry["shape"])
    return keep_masks


@dataclass(frozen=True)
class MaskScopeStats:
    """Sparsity and size reporting for one trained mask."""

    masked_weights: int
    deleted_weights: int
    masked_scope_sparsity: float
    total_model_weights: int | None
    all_weights_sparsity: float | None
    compressed_bytes: int

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def mask_scope_stats(
    keep_masks: Mapping[str, np.ndarray],
    *,
    total_model_weights: int | None = None,
) -> MaskScopeStats:
    """Summarize learned sparsity and on-disk mask size for reporting.

    Args:
        keep_masks: Keep-masks of the trained mask.
        total_model_weights: Total parameter count of the base model, when the
            fraction of *all* weights deleted should be reported (the paper's
            headline number); omit to report only the masked scope.

    Returns:
        Learned sparsity inside the masked scope, optionally over all weights,
        and the compressed artifact size.
    """
    masked_weights = 0
    deleted_weights = 0
    compressed_bytes = 0
    for keep_mask in keep_masks.values():
        masked_weights += int(keep_mask.size)
        deleted_weights += int(keep_mask.size - np.count_nonzero(keep_mask))
        compressed_bytes += len(zstandard.compress(np.packbits(keep_mask.reshape(-1)).tobytes()))
    return MaskScopeStats(
        masked_weights=masked_weights,
        deleted_weights=deleted_weights,
        masked_scope_sparsity=deleted_weights / masked_weights,
        total_model_weights=total_model_weights,
        all_weights_sparsity=(deleted_weights / total_model_weights) if total_model_weights else None,
        compressed_bytes=compressed_bytes,
    )
