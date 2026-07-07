# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared checkpoint loading for the Levanter-based eval entrypoints."""

import jax
import jmp
from haliax import Axis
from haliax.partitioning import ResourceMapping
from levanter.model_loading import load_hf_checkpoint, load_levanter_checkpoint
from levanter.models.lm_model import LmConfig, LmHeadModel


def load_eval_model(
    model_config: LmConfig,
    checkpoint_path: str,
    *,
    checkpoint_is_hf: bool,
    Vocab: Axis,
    axis_mapping: ResourceMapping,
    tokenizer,
    mp: jmp.Policy,
    key: jax.Array,
) -> LmHeadModel:
    """Load an eval model from an HF or Levanter checkpoint.

    ``checkpoint_is_hf`` selects the loader: HF checkpoints resolve the vocab from
    ``tokenizer`` and cast to ``mp.compute_dtype``, while Levanter checkpoints partition
    against the precomputed ``Vocab`` axis. Both loaders must run under an active device mesh.
    """
    if checkpoint_is_hf:
        return load_hf_checkpoint(
            model_config,
            checkpoint_path,
            axis_mapping=axis_mapping,
            tokenizer=tokenizer,
            compute_dtype=mp.compute_dtype,
        )
    return load_levanter_checkpoint(
        model_config,
        checkpoint_path,
        Vocab=Vocab,
        axis_mapping=axis_mapping,
        key=key,
    )
