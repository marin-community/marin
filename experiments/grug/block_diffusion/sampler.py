# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sampling utilities for grug block diffusion (learning experiment).

Training can run without sampling, but a "language model" needs a generation path.

TODO(Learning): implement a block-by-block sampler that:
- takes a prompt (clean prefix)
- generates subsequent blocks sequentially
- within each block, runs an iterative denoising loop for `num_denoise_steps`
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
from jaxtyping import PRNGKeyArray

from experiments.grug.block_diffusion.objective import BlockDiffusionObjectiveConfig, BlockDiffusionTransformer


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float | None = None


def sample_block_diffusion(
    model: BlockDiffusionTransformer,
    prompt_tokens: jax.Array,
    *,
    obj_cfg: BlockDiffusionObjectiveConfig,
    sampling_cfg: SamplingConfig,
    key: PRNGKeyArray,
) -> jax.Array:
    """Generate a continuation using block diffusion.

    Args:
        model: trained block diffusion transformer wrapper.
        prompt_tokens: int array `[S_prompt]` (single example for learning simplicity).

    Returns:
        int array `[S_prompt + <=max_new_tokens]`.

    TODO(Learning):
    - batch this
    - add EOS stopping
    - implement top-p / temperature sampling
    """
    del model, prompt_tokens, obj_cfg, sampling_cfg, key
    raise NotImplementedError("TODO: implement block diffusion sampling")
