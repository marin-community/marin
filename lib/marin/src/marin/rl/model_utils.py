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

import equinox as eqx
import haliax as hax
import jax
from jax.sharding import Mesh
from levanter.checkpoint import load_checkpoint
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint: str | None,
    model_config: LmConfig,
    trainer_config: TrainerConfig,
    vocab_axis: hax.Axis,
    mesh: Mesh | None,
    axis_mapping: dict[str, hax.Axis],
    *,
    key: jax.Array,
) -> LmHeadModel:
    """Load a model from checkpoint, assuming it is in native Levanter format.

    Args:
        checkpoint: Path to checkpoint. If None, builds a new model.
        model_config: Model configuration
        trainer_config: Trainer configuration
        vocab_axis: Vocabulary axis for the model
        mesh: Mesh to shard the model on
        axis_mapping: Axis mapping for parameters
        key: JAX random key for initialization (used if checkpoint is None)

    Returns:
        Loaded model instance
    """
    if checkpoint is None:
        # Build new model from scratch
        return model_config.build(vocab_axis, key=key)

    mp = trainer_config.mp

    # Assume it's a native Levanter checkpoint (msgpack/tensorstore)
    model = eqx.filter_eval_shape(model_config.build, vocab_axis, key=key)
    model = load_checkpoint(model, checkpoint, subpath="model", axis_mapping=axis_mapping, mesh=mesh)
    model = mp.cast_to_compute(model)
    return model
