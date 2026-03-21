# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-pod Levanter training phase for alternating RL.

All hosts run the same entrypoint. Levanter initializes distributed TPU JAX
in the normal way, restores from the latest Levanter checkpoint, and trains
for exactly `steps_per_phase` steps over materialized training batches.

After training completes, the process exports the policy to HF/safetensors
before exiting.
"""

import dataclasses
import logging
import pickle
from dataclasses import dataclass

import haliax as hax
import jax
import jax.random as jrandom
import levanter
from iris.marin_fs import url_to_fs
from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig

from marin.rl.alternating.policy_export import export_policy
from marin.rl.alternating.state import (
    AlternatingRunState,
    MaterializationManifest,
    read_json_from_path,
)
from marin.rl.model_utils import load_model_from_checkpoint
from marin.rl.rl_losses import RLLossModule
from marin.rl.types import TrainingBatch

logger = logging.getLogger(__name__)


@dataclass
class TrainingPhaseConfig:
    """Configuration for one training phase."""

    materialized_manifest_path: str
    run_state_path: str
    model_config: LmConfig
    model_config_class: type[HFCompatConfig]
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    loss: RLLossModule
    tokenizer_name: str
    model_name: str
    initial_checkpoint: str | None = None
    vocab_size: int | None = None
    seed: int = 0
    export_policy_after_training: bool = True
    policy_output_dir: str | None = None


class MaterializedTrainingLoader:
    """Simple finite loader over pre-materialized TrainingBatch files.

    All hosts read the same batch files. Each host calls hax.shard() on the
    deserialized batch to produce the correct per-host shard.
    """

    def __init__(self, batch_paths: list[str], parameter_axis_mapping, mesh):
        self.batch_paths = batch_paths
        self.parameter_axis_mapping = parameter_axis_mapping
        self.mesh = mesh

    def __iter__(self):
        for path in self.batch_paths:
            fs, _ = url_to_fs(path)
            with fs.open(path, "rb") as f:
                batch: TrainingBatch = pickle.load(f)

            # Shard the batch across the full pod mesh
            sharded_batch = _shard_training_batch(batch, self.parameter_axis_mapping, self.mesh)
            yield sharded_batch

    def __len__(self):
        return len(self.batch_paths)


def _shard_training_batch(batch: TrainingBatch, axis_mapping, mesh) -> TrainingBatch:
    """Shard a TrainingBatch across the mesh using hax.shard."""
    with mesh:
        return TrainingBatch(
            input_ids=hax.shard(batch.input_ids, axis_mapping),
            position_ids=hax.shard(batch.position_ids, axis_mapping),
            loss_weights=hax.shard(batch.loss_weights, axis_mapping),
            loss_masks=hax.shard(batch.loss_masks, axis_mapping),
            policy_logprobs=hax.shard(batch.policy_logprobs, axis_mapping),
            temperature=hax.shard(batch.temperature, axis_mapping),
            top_k=hax.shard(batch.top_k, axis_mapping),
            truncated=batch.truncated,
            max_output_tokens=batch.max_output_tokens,
        )


def run_training_phase(config: TrainingPhaseConfig) -> None:
    """Run one training phase over materialized batches.

    1. Initialize Levanter across the full pod
    2. Restore the latest Levanter checkpoint (or initial model)
    3. Train for exactly steps_per_phase steps
    4. Save a new Levanter checkpoint
    5. Export policy to HF/safetensors (if configured)
    """
    # Read manifests
    mat_manifest = MaterializationManifest.from_json(read_json_from_path(config.materialized_manifest_path))
    run_state = AlternatingRunState.from_json(read_json_from_path(config.run_state_path))

    logger.info(
        "Training phase starting: phase_id=%d, policy_version=%d, %d batches",
        run_state.phase_id,
        run_state.policy_version,
        mat_manifest.num_training_batches,
    )

    # Configure checkpointing to point at the right phase directory
    checkpoint_base = run_state.current_levanter_checkpoint_path
    if checkpoint_base is None:
        # First phase: use initial checkpoint if any, otherwise build fresh
        checkpoint_base = config.trainer.checkpointer.base_path

    trainer_config = dataclasses.replace(
        config.trainer,
        checkpointer=dataclasses.replace(
            config.trainer.checkpointer,
            base_path=checkpoint_base,
        ),
    )

    # Initialize Levanter (sets up JAX distributed, mesh, etc.)
    levanter.initialize(trainer_config)

    key = jrandom.PRNGKey(config.seed)
    mesh = trainer_config.device_mesh

    # Build or load model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    vocab_size = config.vocab_size if config.vocab_size is not None else len(tokenizer)
    Vocab = hax.Axis("vocab", vocab_size)

    model = load_model_from_checkpoint(
        checkpoint=config.initial_checkpoint,
        model_config=config.model_config,
        trainer_config=trainer_config,
        vocab_axis=Vocab,
        mesh=mesh,
        axis_mapping=trainer_config.compute_axis_mapping,
        tokenizer=tokenizer,
        key=key,
    )

    # Build reference model for loss function
    reference_model = model
    loss_module = config.loss
    loss_module = loss_module.build(reference_model)
    loss_fn = loss_module.create_loss_fn(reference_model, None)

    @jax.jit
    def _loss_function(model, batch, key):
        return loss_fn(model, batch, key)

    # Build optimizer
    optimizer = config.optimizer.build(trainer_config.num_train_steps)

    # Create trainer and train
    with Trainer(config=trainer_config, optimizer=optimizer, loss_fn=_loss_function) as trainer:
        _, training_key = jrandom.split(jrandom.PRNGKey(trainer_config.seed), 2)
        state = trainer.initial_state(training_key, model=model)

        logger.info("Training state initialized at step %d", int(state.step))

        # Create materialized loader
        loader = MaterializedTrainingLoader(
            batch_paths=mat_manifest.batch_paths,
            parameter_axis_mapping=trainer_config.compute_axis_mapping,
            mesh=mesh,
        )

        # Train
        final_info = trainer.train(state, loader)
        final_step = int(final_info.state.step)
        logger.info("Training phase completed at step %d", final_step)

        # Export policy if configured
        if config.export_policy_after_training and config.policy_output_dir:
            export_policy(
                model=final_info.state.model,
                model_config_class=config.model_config_class,
                tokenizer=tokenizer,
                output_dir=config.policy_output_dir,
                policy_version=run_state.policy_version + 1,
                phase_id=run_state.phase_id,
                source_global_step=final_step,
                model_name=config.model_name,
                levanter_checkpoint_path=checkpoint_base,
            )
            logger.info("Policy exported to %s", config.policy_output_dir)
