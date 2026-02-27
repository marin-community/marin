# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Block diffusion objective for grug (learning experiment).

This experiment aims to train the Grug transformer with a *block diffusion* / masked-denoising
objective instead of standard next-token autoregression.

High-level idea (one reasonable variant):
- Partition a length-S sequence into contiguous blocks of size K.
- Use a *block-causal* attention mask:
  - queries may attend to all keys in earlier blocks
  - queries may attend to all keys in the same block (bidirectional within-block)
  - queries may not attend to future blocks
- For training, pick an "active" block and corrupt some (or all) tokens in that block.
- Train the model to predict the original tokens (x0) at the corrupted positions.

This file is intentionally incomplete: key algorithmic pieces are left as TODOs.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from experiments.grug.base.model import Transformer
from levanter.grug.attention import AttentionMask
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss


@dataclass(frozen=True)
class BlockDiffusionObjectiveConfig:
    """Hyperparameters for the block diffusion objective."""

    block_size: int = 128
    num_denoise_steps: int = 8
    mask_token_id: int = 0


def make_block_causal_mask(*, seq_len: int, block_size: int) -> jax.Array:
    """Return a dense boolean attention mask `[seq_len, seq_len]` implementing block causality.

    Semantics ("allowed" == True):
    - Let block(i) = floor(i / block_size).
    - Query i may attend to key j iff block(j) <= block(i).

    TODO(Learning): implement this efficiently and add a tiny sanity-check script.
    """
    raise NotImplementedError("TODO: implement block-causal attention mask")


def corrupt_for_training(
    tokens: jax.Array,
    *,
    cfg: BlockDiffusionObjectiveConfig,
    key: PRNGKeyArray,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Corrupt a batch for training.

    Args:
        tokens: int array `[B, S]` of clean tokens x0.

    Returns:
        corrupted_tokens: int array `[B, S]` (x_t)
        targets: int array `[B, S]` (usually x0)
        denoise_weight: float array `[B, S]` (1.0 where you score denoising CE, else 0.0)

    TODO(Learning): define your corruption process.
    Typical choices:
    - sample an active block index per example
    - sample a time-step / mask ratio
    - replace selected positions in the active block with cfg.mask_token_id
    - set denoise_weight=1 only on the corrupted positions
    """
    del tokens, cfg, key
    raise NotImplementedError("TODO: implement diffusion corruption + loss weighting")


class BlockDiffusionTransformer(eqx.Module):
    """A `Transformer` wrapper that swaps next-token CE for block diffusion denoising CE."""

    base: Transformer
    obj_cfg: BlockDiffusionObjectiveConfig = eqx.field(static=True)

    def __call__(
        self,
        token_ids: jax.Array,
        *,
        mask: AttentionMask | jax.Array | None = None,
    ) -> jax.Array:
        return self.base(token_ids, mask=mask)

    def compute_next_token_loss(
        self,
        token_ids: jax.Array,
        loss_weight: jax.Array,
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        """Compute the block diffusion denoising loss.

        Note: the name matches the grug template interface; the objective is not next-token.

        TODO(Learning): decide whether you want to combine the dataset's segment IDs/sliding-window
        constraints with block-causality. The code below shows one simple approach.
        """
        if token_ids.ndim != 2:
            raise ValueError(f"Expected token_ids [B,S], got shape={token_ids.shape}")
        if loss_weight.shape != token_ids.shape:
            raise ValueError(f"loss_weight shape {loss_weight.shape} must match tokens {token_ids.shape}")

        if key is None:
            # Deterministic fallback for eval-like contexts.
            key = jax.random.PRNGKey(0)

        seq_len = int(token_ids.shape[1])
        block_mask = make_block_causal_mask(seq_len=seq_len, block_size=self.obj_cfg.block_size)

        # Preserve segment IDs / sliding windows from the input mask, but drop token-level causality.
        if isinstance(mask, AttentionMask):
            seg_only = AttentionMask(is_causal=False, segment_ids=mask.segment_ids, sliding_window=mask.sliding_window)
            seg_allowed = seg_only.materialize_mask(seq_len, seq_len)
            if seg_allowed is not None:
                block_mask = jnp.logical_and(block_mask, seg_allowed)

        corrupted, targets, denoise_weight = corrupt_for_training(token_ids, cfg=self.obj_cfg, key=key)

        denoise_weight = denoise_weight.astype(loss_dtype) * loss_weight.astype(loss_dtype)
        targets = targets.astype(jnp.int32)

        hidden = self.base(corrupted, mask=block_mask)

        return fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.base.output_proj,
            targets,
            weight=denoise_weight,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
        )
