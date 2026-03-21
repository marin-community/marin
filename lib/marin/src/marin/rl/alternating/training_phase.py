# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training-phase entrypoints for alternating RL."""

from __future__ import annotations

import dataclasses
import logging

import haliax as hax
import jax
import jax.random as jrandom
import levanter
from haliax import Axis
from levanter.checkpoint import discover_latest_checkpoint
from levanter.main.export_lm_to_hf import ConvertLmConfig
from levanter.main import export_lm_to_hf
from levanter.trainer import Trainer
from levanter.utils.jax_utils import barrier_sync
from transformers import AutoTokenizer

from marin.rl.alternating.config import AlternatingRLConfig
from marin.rl.alternating.io import read_pickle
from marin.rl.alternating.state import (
    AlternatingRunPaths,
    AlternatingRunState,
    MaterializedBatchesManifest,
    PolicyManifest,
    utc_now_iso,
    write_policy_manifest,
)
from marin.rl.model_utils import load_model_from_checkpoint

logger = logging.getLogger(__name__)


class MaterializedTrainingLoader:
    """Finite loader over materialized `TrainingBatch` pickles."""

    def __init__(self, batch_paths: list[str], trainer_config):
        self.batch_paths = batch_paths
        self.trainer_config = trainer_config

    def __iter__(self):
        for batch_path in self.batch_paths:
            batch = read_pickle(batch_path)
            with hax.set_mesh(self.trainer_config.device_mesh):
                yield hax.shard(batch, self.trainer_config.compute_axis_mapping)


def _trainer_run_id(config: AlternatingRLConfig) -> str:
    return f"{config.run_id}-alternating-train"


def _trainer_config_for_phase(
    config: AlternatingRLConfig,
    paths: AlternatingRunPaths,
    *,
    total_train_steps: int,
    resume_checkpoint_path: str | None,
):
    checkpointer = dataclasses.replace(
        config.trainer.checkpointer,
        base_path=paths.levanter_checkpoints_root,
    )
    trainer_run_id = _trainer_run_id(config)
    return dataclasses.replace(
        config.trainer,
        id=trainer_run_id,
        num_train_steps=total_train_steps,
        load_checkpoint=resume_checkpoint_path is not None,
        load_checkpoint_path=resume_checkpoint_path,
        checkpointer=checkpointer,
    )


def _build_policy_model(config: AlternatingRLConfig, trainer_config, tokenizer):
    vocab_size = config.vocab_size if config.vocab_size is not None else len(tokenizer)
    vocab_axis = Axis("vocab", vocab_size)
    return load_model_from_checkpoint(
        checkpoint=config.initial_checkpoint,
        model_config=config.model,
        trainer_config=trainer_config,
        vocab_axis=vocab_axis,
        tokenizer=tokenizer,
        mesh=trainer_config.device_mesh,
        axis_mapping=trainer_config.parameter_axis_mapping,
        key=jrandom.PRNGKey(config.seed),
    )


def _trainer_checkpoint_root(trainer_config) -> str:
    return trainer_config.checkpointer.expanded_path(trainer_config.id)


def run_training_phase(
    config: AlternatingRLConfig,
    paths: AlternatingRunPaths,
    state: AlternatingRunState,
    manifest: MaterializedBatchesManifest,
) -> str:
    """Run the Levanter training phase and export the next sampling policy."""
    remaining_steps = config.quotas.num_train_steps - state.source_global_step
    if remaining_steps <= 0:
        raise ValueError("training phase requested with no remaining steps")

    batch_paths = manifest.batch_paths[: min(len(manifest.batch_paths), remaining_steps)]
    if not batch_paths:
        raise ValueError(f"materialized manifest for phase {state.phase_id} has no training batches")
    target_step = state.source_global_step + len(batch_paths)
    trainer_config = _trainer_config_for_phase(
        config,
        paths,
        total_train_steps=target_step,
        resume_checkpoint_path=state.current_levanter_checkpoint_path,
    )
    trainer_checkpoint_root = _trainer_checkpoint_root(trainer_config)

    levanter.initialize(trainer_config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    reference_model = _build_policy_model(config, trainer_config, tokenizer)
    optimizer = config.optimizer.build(config.quotas.num_train_steps)  # type: ignore[assignment]
    loss_fn = config.loss.create_loss_fn(reference_model, None)

    @jax.jit
    def _loss_function(model, batch, key):
        return loss_fn(model, batch, key)

    loader = MaterializedTrainingLoader(batch_paths, trainer_config)
    with Trainer(config=trainer_config, optimizer=optimizer, loss_fn=_loss_function) as trainer:
        training_key = jrandom.PRNGKey(trainer_config.seed)
        initial_state = trainer.initial_state(training_key, model=reference_model)
        trainer.train(initial_state, loader)

    barrier_sync()
    latest_checkpoint = discover_latest_checkpoint(trainer_checkpoint_root)
    if latest_checkpoint is None:
        raise FileNotFoundError(f"training phase completed without a checkpoint under {trainer_checkpoint_root}")

    next_policy_version = manifest.policy_version + 1
    export_dir = paths.policy_dir(next_policy_version)
    convert_config = ConvertLmConfig(
        trainer=trainer_config,
        checkpoint_path=latest_checkpoint,
        output_dir=export_dir,
        model=config.model,
        tokenizer=config.tokenizer_name,
        use_cpu=True,
    )

    if jax.process_index() == 0:
        # HF conversion currently assumes a single-process export path, so rank 0
        # reloads the checkpoint on CPU and writes the immutable sampling policy.
        export_lm_to_hf.main(convert_config)
        policy_manifest_path = paths.policy_manifest_path(next_policy_version)
        write_policy_manifest(
            policy_manifest_path,
            PolicyManifest(
                policy_version=next_policy_version,
                phase_id=state.phase_id,
                source_global_step=target_step,
                policy_path=export_dir,
                levanter_checkpoint_path=latest_checkpoint,
                model_name=config.inference.model_name,
                tokenizer_name=config.tokenizer_name,
                enable_fast_bootstrap=True,
                created_at=utc_now_iso(),
            ),
        )
    barrier_sync()
    return paths.policy_manifest_path(next_policy_version)
