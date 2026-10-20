# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a native sweep checkpoint without context parallelism."""

import json
import logging
from dataclasses import dataclass
from functools import partial

import jax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter import config as levanter_config
from levanter.data.text.datasets import LmDataConfig
from levanter.eval import TaggedEvaluator, _default_lm_eval_loss_fn, eval_model
from levanter.model_loading import load_levanter_checkpoint
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig, initialize
from levanter.utils.mesh import MeshConfig
from rigging.filesystem.storage_path import StoragePath

logger = logging.getLogger(__name__)

OFFLINE_MESH = MeshConfig(
    axes={"data": 1, "replica": 1, "model": 1, "expert": 8},
    dcn_axes={"replica_dcn": 1},
    compute_mapping={"batch": ["replica_dcn", "data", "expert"], "vocab": "model"},
)


@dataclass
class OfflineValidationConfig:
    checkpoint_path: str
    output_path: str
    model: LmConfig
    data: LmDataConfig
    trainer: TrainerConfig
    sequence_length: int


def evaluate_checkpoint(config: OfflineValidationConfig) -> dict:
    """Write token-weighted family and pooled losses from one native checkpoint."""
    initialize(config)
    mapping = config.trainer.compute_axis_mapping
    if mapping.get("position") is not None:
        raise ValueError("Offline validation requires an unsharded position axis")
    tokenizer = config.data.the_tokenizer
    datasets = config.data.tagged_eval_sets(Axis("position", config.sequence_length))
    if not datasets:
        raise ValueError("Offline validation requires held-out datasets")
    with config.trainer.use_device_mesh():
        vocab = round_axis_for_partitioning(Axis("vocab", len(tokenizer)), mapping)
        model = load_levanter_checkpoint(
            config.model,
            config.checkpoint_path,
            Vocab=vocab,
            axis_mapping=config.trainer.parameter_axis_mapping,
            key=jax.random.PRNGKey(0),
        )
        evaluator = TaggedEvaluator(
            EvalBatch=config.trainer.EvalBatch,
            tagged_eval_sets=datasets,
            loss_fn=partial(
                _default_lm_eval_loss_fn,
                EvalBatch=config.trainer.EvalBatch,
                mp=config.trainer.mp,
            ),
            tokenizer=tokenizer,
            device_mesh=config.trainer.device_mesh,
            axis_mapping=mapping,
            max_examples_per_dataset=None,
        )
        metrics = {key: float(value) for key, value in eval_model(evaluator, model, prefix="eval").items()}
    result = {"checkpoint_path": config.checkpoint_path, "metrics": metrics}
    if jax.process_index() == 0:
        StoragePath(config.output_path).write_text(json.dumps(result, sort_keys=True) + "\n")
        logger.info("Offline validation result: %s", json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    levanter_config.main(evaluate_checkpoint)()
