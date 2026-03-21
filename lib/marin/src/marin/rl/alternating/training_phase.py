# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training-phase entrypoints for alternating RL."""

from __future__ import annotations

import dataclasses
import json
import logging
import time

import haliax as hax
import jax
import jax.random as jrandom
import levanter
from levanter.utils.fsspec_utils import exists
from haliax import Axis
from levanter.checkpoint import discover_latest_checkpoint
from levanter.trainer import Trainer
from levanter.utils.jax_utils import barrier_sync
from iris.marin_fs import url_to_fs
from transformers import AutoTokenizer

from marin.rl.alternating.config import AlternatingRLConfig
from marin.rl.alternating.io import read_pickle
from marin.rl.alternating.state import (
    AlternatingRunPaths,
    AlternatingRunState,
    MaterializedBatchesManifest,
    PhaseMetricsManifest,
    PolicyBootstrapFormat,
    PolicyManifest,
    RunStatus,
    update_phase_metrics,
    utc_now_iso,
    write_run_state,
    write_policy_manifest,
)
from marin.rl.alternating.wandb import alternating_training_tracker_config
from marin.rl.model_utils import load_model_from_checkpoint
from marin.rl.training_observability import (
    TrainingSamplePreview,
    configure_rl_training_metric_hooks,
    maybe_read_training_sample_previews,
    training_sample_preview_path,
)

logger = logging.getLogger(__name__)

CHECKPOINT_VISIBILITY_TIMEOUT = 180
CHECKPOINT_VISIBILITY_POLL_INTERVAL = 5


class MaterializedTrainingLoader:
    """Finite loader over materialized `TrainingBatch` pickles."""

    def __init__(self, batch_paths: list[str], trainer_config):
        self.batch_paths = batch_paths
        self.trainer_config = trainer_config
        self._last_batch_prep_time: float = 0.0
        self._last_sample_previews: list[TrainingSamplePreview] | None = None

    def __iter__(self):
        for batch_path in self.batch_paths:
            prep_start = time.time()
            batch = read_pickle(batch_path)
            preview_path = training_sample_preview_path(batch_path)
            self._last_sample_previews = (
                maybe_read_training_sample_previews(preview_path) if exists(preview_path) else None
            )
            with hax.set_mesh(self.trainer_config.device_mesh):
                sharded_batch = hax.shard(batch, self.trainer_config.compute_axis_mapping)
            self._last_batch_prep_time = time.time() - prep_start
            yield sharded_batch


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
    replace_kwargs = dict(
        id=trainer_run_id,
        num_train_steps=total_train_steps,
        load_checkpoint=resume_checkpoint_path is not None,
        load_checkpoint_path=resume_checkpoint_path,
        checkpointer=checkpointer,
    )
    if hasattr(config.trainer, "tracker"):
        replace_kwargs["tracker"] = alternating_training_tracker_config(
            getattr(config.trainer, "tracker", None),
            run_id=config.run_id,
        )
    return dataclasses.replace(config.trainer, **replace_kwargs)


def _trainer_checkpoint_root(trainer_config) -> str:
    return trainer_config.checkpointer.expanded_path(trainer_config.id)


def _reference_checkpoint_for_phase(
    config: AlternatingRLConfig,
    *,
    resume_checkpoint_path: str | None,
) -> str | None:
    if resume_checkpoint_path is None:
        return config.initial_checkpoint

    # When the loss has no active KL/reference-model term, resumed phases only
    # need a correctly-shaped model tree. The trainer checkpoint restore will
    # overwrite the weights immediately, so we can skip reloading the original HF
    # checkpoint on every phase boundary.
    kl_coef = getattr(config.loss, "kl_coef", None)
    if kl_coef is not None and float(kl_coef) == 0.0:
        return None

    return config.initial_checkpoint


def _build_reference_model(
    config: AlternatingRLConfig,
    trainer_config,
    tokenizer,
    *,
    resume_checkpoint_path: str | None,
):
    checkpoint = _reference_checkpoint_for_phase(
        config,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    vocab_size = config.vocab_size if config.vocab_size is not None else len(tokenizer)
    vocab_axis = Axis("vocab", vocab_size)
    return load_model_from_checkpoint(
        checkpoint=checkpoint,
        model_config=config.model,
        trainer_config=trainer_config,
        vocab_axis=vocab_axis,
        tokenizer=tokenizer,
        mesh=trainer_config.device_mesh,
        axis_mapping=trainer_config.parameter_axis_mapping,
        key=jrandom.PRNGKey(config.seed),
    )


def _phase_batch_paths(
    config: AlternatingRLConfig,
    state: AlternatingRunState,
    manifest: MaterializedBatchesManifest,
) -> tuple[list[str], int]:
    remaining_steps = config.quotas.num_train_steps - state.source_global_step
    if remaining_steps <= 0:
        raise ValueError("training phase requested with no remaining steps")

    batch_paths = manifest.batch_paths[: min(len(manifest.batch_paths), remaining_steps)]
    if not batch_paths:
        raise ValueError(f"materialized manifest for phase {state.phase_id} has no training batches")

    return batch_paths, state.source_global_step + len(batch_paths)


def _checkpoint_step(checkpoint_path: str) -> int:
    fs, fs_path = url_to_fs(checkpoint_path)
    with fs.open(f"{fs_path}/metadata.json", "rt", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return int(metadata["step"])


def _checkpoint_metadata_exists(checkpoint_path: str) -> bool:
    fs, fs_path = url_to_fs(checkpoint_path, use_listings_cache=False)
    return fs.exists(f"{fs_path}/metadata.json")


def _wait_for_target_checkpoint(
    checkpoint_root: str,
    *,
    target_step: int,
    timeout: int = CHECKPOINT_VISIBILITY_TIMEOUT,
    poll_interval: int = CHECKPOINT_VISIBILITY_POLL_INTERVAL,
) -> str:
    expected_checkpoint = f"{checkpoint_root}/step-{target_step}"
    deadline = time.monotonic() + timeout
    last_seen_checkpoint: str | None = None
    last_seen_step: int | None = None

    while True:
        if _checkpoint_metadata_exists(expected_checkpoint):
            return expected_checkpoint

        latest_checkpoint = discover_latest_checkpoint(checkpoint_root)
        if latest_checkpoint is not None:
            last_seen_checkpoint = latest_checkpoint
            last_seen_step = _checkpoint_step(latest_checkpoint)
            if last_seen_step >= target_step:
                return latest_checkpoint

        if time.monotonic() >= deadline:
            raise FileNotFoundError(
                "training phase completed without a visible checkpoint at the expected step: "
                f"checkpoint_root={checkpoint_root}, target_step={target_step}, "
                f"last_seen_checkpoint={last_seen_checkpoint}, last_seen_step={last_seen_step}"
            )

        time.sleep(poll_interval)


def _log_phase_metrics_to_tracker(
    phase_metrics: PhaseMetricsManifest,
    *,
    phase_id: int,
    source_global_step: int,
    tracker=None,
) -> None:
    metrics = {
        "alternating/phase_id": phase_id,
        "alternating/source_global_step": source_global_step,
        "alternating/phase_total_seconds": phase_metrics.total_recorded_seconds,
    }
    if phase_metrics.prepare_sampling_seconds is not None:
        metrics["alternating/prepare_sampling_seconds"] = phase_metrics.prepare_sampling_seconds
    if phase_metrics.sampling_seconds is not None:
        metrics["alternating/sampling_seconds"] = phase_metrics.sampling_seconds
    if phase_metrics.curriculum_update_seconds is not None:
        metrics["alternating/curriculum_update_seconds"] = phase_metrics.curriculum_update_seconds
    if phase_metrics.materialization_seconds is not None:
        metrics["alternating/materialization_seconds"] = phase_metrics.materialization_seconds
    if phase_metrics.training_seconds is not None:
        metrics["alternating/training_seconds"] = phase_metrics.training_seconds
    if phase_metrics.export_seconds is not None:
        metrics["alternating/export_seconds"] = phase_metrics.export_seconds

    if tracker is None:
        try:
            tracker = levanter.tracker.current_tracker()
        except RuntimeError:
            logger.info("No active tracker for phase %d; skipping alternating phase metric log", phase_id)
            return

    tracker.log(metrics, step=source_global_step)
    logger.info("alternating phase metrics: %s", metrics)


def _publish_policy_manifest(
    config: AlternatingRLConfig,
    paths: AlternatingRunPaths,
    *,
    phase_id: int,
    source_global_step: int,
    policy_version: int,
    checkpoint_path: str,
) -> str:
    if jax.process_index() == 0:
        policy_manifest_path = paths.policy_manifest_path(policy_version)
        write_policy_manifest(
            policy_manifest_path,
            PolicyManifest(
                policy_version=policy_version,
                phase_id=phase_id,
                source_global_step=source_global_step,
                policy_path=checkpoint_path,
                levanter_checkpoint_path=checkpoint_path,
                model_name=config.inference.model_name,
                tokenizer_name=config.tokenizer_name,
                enable_fast_bootstrap=True,
                created_at=utc_now_iso(),
                bootstrap_format=PolicyBootstrapFormat.LEVANTER_CHECKPOINT,
            ),
        )
    barrier_sync()
    return paths.policy_manifest_path(policy_version)


def export_policy_only(
    config: AlternatingRLConfig,
    paths: AlternatingRunPaths,
    state: AlternatingRunState,
    manifest: MaterializedBatchesManifest,
) -> str:
    """Publish the next policy manifest from an already-completed training checkpoint."""

    def _recover_run_state(checkpoint_path: str | None) -> None:
        if jax.process_index() != 0:
            return
        recovered_state = dataclasses.replace(
            state,
            status=RunStatus.TRAINING,
            current_levanter_checkpoint_path=checkpoint_path,
        )
        write_run_state(paths.run_state_path, recovered_state)

    expected_policy_manifest_path = paths.policy_manifest_path(state.policy_version + 1)
    policy_manifest_fs, policy_manifest_fs_path = url_to_fs(expected_policy_manifest_path)
    if policy_manifest_fs.exists(policy_manifest_fs_path):
        policy_manifest = PolicyManifest.from_dict(json.loads(policy_manifest_fs.cat(policy_manifest_fs_path).decode()))
        _recover_run_state(policy_manifest.levanter_checkpoint_path)
        return expected_policy_manifest_path

    batch_paths, target_step = _phase_batch_paths(config, state, manifest)
    del batch_paths
    trainer_config = _trainer_config_for_phase(
        config,
        paths,
        total_train_steps=target_step,
        resume_checkpoint_path=state.current_levanter_checkpoint_path,
    )
    trainer_checkpoint_root = _trainer_checkpoint_root(trainer_config)
    latest_checkpoint = discover_latest_checkpoint(trainer_checkpoint_root)
    if latest_checkpoint is None:
        raise FileNotFoundError(f"export-only recovery found no checkpoint under {trainer_checkpoint_root}")

    checkpoint_step = _checkpoint_step(latest_checkpoint)
    if checkpoint_step != target_step:
        raise ValueError(
            "export-only recovery refuses to publish from an unexpected checkpoint step: "
            f"have={checkpoint_step}, expected={target_step}"
        )

    export_start = time.time()
    policy_manifest_path = _publish_policy_manifest(
        config,
        paths,
        phase_id=state.phase_id,
        source_global_step=target_step,
        policy_version=manifest.policy_version + 1,
        checkpoint_path=latest_checkpoint,
    )
    phase_metrics = update_phase_metrics(
        paths.phase_metrics_path(state.phase_id),
        phase_id=state.phase_id,
        export_seconds=time.time() - export_start,
    )
    if jax.process_index() == 0:
        _log_phase_metrics_to_tracker(phase_metrics, phase_id=state.phase_id, source_global_step=target_step)
        _recover_run_state(latest_checkpoint)
    return policy_manifest_path


def run_training_phase(
    config: AlternatingRLConfig,
    paths: AlternatingRunPaths,
    state: AlternatingRunState,
    manifest: MaterializedBatchesManifest,
) -> str:
    """Run the Levanter training phase and export the next sampling policy."""
    batch_paths, target_step = _phase_batch_paths(config, state, manifest)
    trainer_config = _trainer_config_for_phase(
        config,
        paths,
        total_train_steps=target_step,
        resume_checkpoint_path=state.current_levanter_checkpoint_path,
    )
    trainer_checkpoint_root = _trainer_checkpoint_root(trainer_config)

    levanter.initialize(trainer_config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    reference_model = _build_reference_model(
        config,
        trainer_config,
        tokenizer,
        resume_checkpoint_path=state.current_levanter_checkpoint_path,
    )
    optimizer = config.optimizer.build(config.quotas.num_train_steps)  # type: ignore[assignment]
    loss_fn = config.loss.create_loss_fn(reference_model, None)
    vocab_size = config.vocab_size if config.vocab_size is not None else len(tokenizer)
    tokens_per_example = config.curriculum.max_seq_len
    flops_per_token = config.model.flops_per_token(vocab_size, tokens_per_example)
    flops_per_example = 3 * flops_per_token * tokens_per_example if flops_per_token is not None else None

    @jax.jit
    def _loss_function(model, batch, key):
        return loss_fn(model, batch, key)

    loader = MaterializedTrainingLoader(batch_paths, trainer_config)
    tracker = None
    with Trainer(config=trainer_config, optimizer=optimizer, loss_fn=_loss_function) as trainer:
        training_key = jrandom.PRNGKey(trainer_config.seed)
        initial_state = trainer.initial_state(training_key, model=reference_model)
        configure_rl_training_metric_hooks(
            trainer,
            tokenizer=tokenizer,
            tokens_per_example=tokens_per_example,
            flops_per_example=flops_per_example,
            batch_schedule=trainer_config.train_batch_size,
            batch_prep_time=lambda: loader._last_batch_prep_time,
            sample_previews=lambda: loader._last_sample_previews,
        )
        training_start = time.time()
        trainer.train(initial_state, loader)
        tracker = getattr(trainer, "tracker", None)
    training_seconds = time.time() - training_start
    phase_metrics = update_phase_metrics(
        paths.phase_metrics_path(state.phase_id),
        phase_id=state.phase_id,
        training_seconds=training_seconds,
    )

    barrier_sync()
    latest_checkpoint = _wait_for_target_checkpoint(
        trainer_checkpoint_root,
        target_step=target_step,
    )

    export_start = time.time()
    policy_manifest_path = _publish_policy_manifest(
        config,
        paths,
        phase_id=state.phase_id,
        source_global_step=target_step,
        policy_version=manifest.policy_version + 1,
        checkpoint_path=latest_checkpoint,
    )
    phase_metrics = update_phase_metrics(
        paths.phase_metrics_path(state.phase_id),
        phase_id=state.phase_id,
        export_seconds=time.time() - export_start,
    )
    if jax.process_index() == 0:
        _log_phase_metrics_to_tracker(
            phase_metrics,
            phase_id=state.phase_id,
            source_global_step=target_step,
            tracker=tracker,
        )
    if tracker is not None:
        tracker.finish()
    return policy_manifest_path
