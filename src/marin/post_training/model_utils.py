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
Training worker for RL/post-training tasks.

This worker reads rollout information from a queue which is populated by the
rollout workers, and periodically dumps new checkpoints to disk. These
checkpoints are read by the rollout workers to update their models.
"""

import logging

import equinox as eqx
import haliax as hax
import jax
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device

logger = logging.getLogger(__name__)


def load_model_from_checkpoint_or_hf(
    model_config: LmConfig,
    trainer_config: TrainerConfig,
    vocab_axis: hax.Axis,
    tokenizer,
    *,
    levanter_checkpoint_path: str | None = None,
    hf_checkpoint: RepoRef | None = None,
    key: jax.Array,
) -> LmHeadModel:
    """Load a model either from a checkpoint or HF repo."""
    if levanter_checkpoint_path is None and hf_checkpoint is None:
        raise ValueError("Must specify either checkpoint_path or hf_checkpoint")
    if levanter_checkpoint_path is not None and hf_checkpoint is not None:
        raise ValueError("Specify only one of checkpoint_path or hf_checkpoint")

    mp = trainer_config.mp

    if levanter_checkpoint_path is not None:
        with use_cpu_device():
            model = eqx.filter_eval_shape(model_config.build, vocab_axis, key=key)
            model = load_checkpoint(model, levanter_checkpoint_path, subpath="model")
            model = mp.cast_to_compute(model)
        return model

    if not hasattr(model_config, "hf_checkpoint_converter"):
        raise ValueError("Model config lacks HF checkpoint converter for loading from HuggingFace")

    converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
    converter = converter.replaced(reference_checkpoint=hf_checkpoint, tokenizer=tokenizer)
    model = converter.load_pretrained(model_config.model_type, ref=hf_checkpoint, dtype=trainer_config.mp.compute_dtype)
    return model
