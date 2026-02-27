"""Block diffusion objective for Grugformer (learning experiment).

This file is intentionally incomplete: it scaffolds the integration points and
leaves the core algorithmic pieces as TODOs.

Goal: replace next-token autoregressive loss with a block diffusion denoising
loss while reusing Grugformer (`levanter.grug`) for the transformer.

Key constraints from current Grug core:
- `levanter.grug.model.activations` accepts either a structured `AttentionMask` or
  a dense JAX array mask.
- On TPU, passing a dense mask forces the reference attention path
  (`levanter.grug.attention.attention`). See `lib/levanter/src/levanter/grug/attention.py:379`.

If you want TPU-performance later, you'll likely need to extend Grug's
`AttentionMask` + Splash integration, but that is a second-phase exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from levanter.grug.model import activations as grug_activations
from levanter.models.grug_wrapper import GrugWrapper
from levanter.models.lm_model import LmExample, LmHeadModel


@dataclass(frozen=True)
class BlockDiffusionObjectiveConfig:
    """Configuration for a semi-autoregressive block diffusion LM."""

    block_size: int = 128
    """Block size K (contiguous tokens). Keep `max_seq_len % block_size == 0`."""

    num_denoise_steps: int = 8
    """How many denoising iterations to run *within* a block at sampling time."""

    mask_token_id: int = 0
    """Sentinel token id used for masking/corruption.

    Learning exercise: decide whether you want a true `[MASK]` token (requires
    tokenizer changes) or a different corruption process (e.g., random token
    replacement) that doesn't require a new vocab token.
    """

    # TODO: add schedule parameters (e.g. cosine schedule) once you implement time conditioning.


def make_block_causal_mask(*, seq_len: int, block_size: int) -> jax.Array:
    """Return a dense attention mask implementing block-causal semantics.

    Required semantics:
    - Query tokens can attend to all keys in *earlier blocks*.
    - Query tokens can attend to all keys in the *same block* (bidirectional inside-block).
    - Query tokens cannot attend to keys in *future blocks*.

    Returns:
        A boolean mask shaped `[seq_len, seq_len]` where True means "allowed".

    TODO(Learning): implement efficiently with JAX ops and validate it on a toy example.
    """
    raise NotImplementedError("TODO: implement block-causal attention mask")


def corrupt_for_training(
    tokens: jax.Array,
    *,
    cfg: BlockDiffusionObjectiveConfig,
    key: PRNGKeyArray,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Create a corrupted training input for block diffusion.

    Inputs:
        tokens: int32 array `[B, S]` of clean tokens (x0).

    Returns:
        corrupted_tokens: int32 `[B, S]` (x_t with prefix conditioning + masked current block)
        targets: int32 `[B, S]` (usually x0; you may choose to only score a block)
        loss_weight: float32 `[B, S]` (1 for positions you want to score; 0 elsewhere)
        attn_mask: boolean `[S, S]` or `[B, S, S]` (block-causal)

    TODO(Learning): define
    - how you pick the active block index b
    - how you pick the diffusion time t / mask ratio
    - whether you mask future blocks to match the block-factorized generation story
    """
    raise NotImplementedError("TODO: implement the corruption process + loss weights")


class GrugBlockDiffusionWrapper(LmHeadModel[Any]):
    """LmHeadModel wrapper that trains Grugformer as a block diffusion LM.

    This wrapper should:
    - keep the same parameterization as Grugformer (plus optional extra params)
    - override `compute_next_token_loss` to compute the denoising objective

    TODO(Learning): add time-step conditioning. Minimal options:
    - add a learned embedding for t and add it to token embeddings
    - or use a small MLP on a scalar noise level and add to hidden states

    Keep this experiment-scoped until you have tests + a working run.
    """

    grug: GrugWrapper
    obj_cfg: BlockDiffusionObjectiveConfig = eqx.field(static=True)

    @property
    def config(self):  # type: ignore[override]
        return self.grug.config

    @property
    def Pos(self):  # type: ignore[override]
        return self.grug.Pos

    @property
    def KeyPos(self):  # type: ignore[override]
        return self.grug.KeyPos

    @property
    def Vocab(self):  # type: ignore[override]
        return self.grug.Vocab

    @property
    def Embed(self):  # type: ignore[override]
        return self.grug.Embed

    def activations(self, input_ids, attn_mask=None, *, key=None, pos_ids=None):  # type: ignore[override]
        return self.grug.activations(input_ids, attn_mask, key=key, pos_ids=pos_ids)

    def get_lm_head(self):  # type: ignore[override]
        return self.grug.get_lm_head()

    def resize_vocab(self, new_size: int, key: PRNGKeyArray | None = None):  # type: ignore[override]
        return cast(Any, self.grug.resize_vocab(new_size, key=key))

    def compute_next_token_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction: hax.ReductionFunction | None = cast(hax.ReductionFunction | None, hax.mean),
        reduction_axis: hax.AxisSelection | None = None,
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype | None = jnp.float32,
        logit_soft_cap: float | None = None,
    ):
        # NOTE: trainer passes `key` (PRNG) here; use it to sample corruption.
        del reduction_axis, logsumexp_weight, loss_dtype
        assert logit_soft_cap is None, "logit_soft_cap not wired for this experiment"

        tokens = example.tokens.array

        # TODO(Learning): Decide what "valid positions" means for diffusion.
        # `example.loss_weight` is causal by default in Levanter; for diffusion you probably want
        # 1s everywhere except padding/ignore ids.
        valid_weight = example.loss_weight.array.astype(jnp.float32)

        if key is None:
            # Deterministic fallback for eval-only contexts.
            key = jax.random.key(0)

        corrupted, targets, loss_weight, attn_mask = corrupt_for_training(tokens, cfg=self.obj_cfg, key=key)
        loss_weight = loss_weight * valid_weight

        # TODO(Learning): if you add timestep conditioning, thread it into the forward pass here.
        hidden = grug_activations(self.grug.params, corrupted, self.grug.grug_config, mask=attn_mask)

        # TODO(Learning): compute CE only on `loss_weight==1` positions.
        # Hint: reuse `levanter.grug.loss.fused_linear_softmax_cross_entropy_loss` to avoid materializing full logits.
        raise NotImplementedError("TODO: compute fused CE denoising loss")
