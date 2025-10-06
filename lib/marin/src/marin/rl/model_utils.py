# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model utilities for RL/post-training tasks.

Contains helper functions for loading models from various checkpoint formats,
including both local Levanter checkpoints and HuggingFace repositories.
"""

import logging
from contextlib import ExitStack

import equinox as eqx
import haliax as hax
import jax
from jax.sharding import Mesh
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig

logger = logging.getLogger(__name__)


def is_hf_checkpoint(checkpoint: str) -> bool:
    """Determine if checkpoint is a HuggingFace model or local path.

    Uses a simple heuristic: if the checkpoint looks like a path then
    assume it is a local Levanter checkpoint; otherwise assume it is a
    HuggingFace repository.
    """
    return not (
        "://" in checkpoint or checkpoint.startswith("/") or checkpoint.startswith("./") or checkpoint.startswith("../")
    )


def load_model_from_checkpoint(
    checkpoint: str | None,
    model_config: LmConfig,
    trainer_config: TrainerConfig,
    vocab_axis: hax.Axis,
    mesh: Mesh | None,
    axis_mapping: dict[str, hax.Axis],
    tokenizer,
    *,
    key: jax.Array,
) -> LmHeadModel:
    """Load a model from checkpoint, auto-detecting HF vs local Levanter format.

    Args:
        checkpoint: Path to checkpoint. If None, builds a new model.
                   Auto-detects HF repo vs local path using heuristics.
        model_config: Model configuration
        trainer_config: Trainer configuration
        vocab_axis: Vocabulary axis for the model
        tokenizer: Tokenizer instance
        key: JAX random key for initialization

    Returns:
        Loaded model instance
    """
    if checkpoint is None:
        # Build new model from scratch
        return model_config.build(vocab_axis, key=key)

    mp = trainer_config.mp

    if is_hf_checkpoint(checkpoint):
        # Load from HuggingFace
        if not hasattr(model_config, "hf_checkpoint_converter"):
            raise ValueError("Model config lacks HF checkpoint converter for loading from HuggingFace")

        hf_checkpoint = RepoRef.from_string(checkpoint)
        converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
        converter = converter.replaced(reference_checkpoint=hf_checkpoint, tokenizer=tokenizer)
        with ExitStack() as stack:
            if mesh is not None:
                stack.enter_context(mesh)

            model = converter.load_pretrained(
                model_config.model_type,
                ref=hf_checkpoint,
                config=model_config,
                axis_mapping=axis_mapping,
                dtype=trainer_config.mp.compute_dtype,
            )
        return model
    else:
        # Load from local Levanter checkpoint
        model = eqx.filter_eval_shape(model_config.build, vocab_axis, key=key)
        model = load_checkpoint(model, checkpoint, subpath="model", axis_mapping=axis_mapping, mesh=mesh)
        model = mp.cast_to_compute(model)
        return model
