"""Sampling utilities for the block diffusion experiment.

This is intentionally a stub: training can run without sampling, but an end-to-end
"language model" needs a generation path.

TODO(Learning): implement a block-by-block sampler that:
- consumes a prompt (clean prefix)
- generates subsequent blocks sequentially
- within each block, runs an iterative denoising loop for `num_denoise_steps`
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from experiments.speedrun.grugformer_block_diffusion.objective import BlockDiffusionObjectiveConfig, GrugBlockDiffusionWrapper


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float | None = None


def sample_block_diffusion(
    model: GrugBlockDiffusionWrapper,
    prompt_tokens: jax.Array,
    *,
    obj_cfg: BlockDiffusionObjectiveConfig,
    sampling_cfg: SamplingConfig,
    key: PRNGKeyArray,
) -> jax.Array:
    """Generate a continuation using block diffusion.

    Args:
        model: trained diffusion wrapper.
        prompt_tokens: int32 array `[S_prompt]` (single example for learning simplicity).

    Returns:
        int32 array `[S_prompt + <=max_new_tokens]`.

    TODO(Learning): make this batched + integrate EOS stopping.
    """
    del model, prompt_tokens, obj_cfg, sampling_cfg, key
    raise NotImplementedError("TODO: implement iterative denoising per block")
