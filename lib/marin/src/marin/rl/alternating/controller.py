# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller loop for alternating single-pod RL."""

from __future__ import annotations

import traceback
import logging
from dataclasses import replace
from typing import Protocol

from levanter.utils.fsspec_utils import exists

from marin.rl.alternating.config import AlternatingRLConfig
from marin.rl.alternating.state import (
    AlternatingRunPaths,
    AlternatingRunState,
    HostPhaseStatus,
    MaterializedBatchesManifest,
    RunStatus,
    SamplingHostAssignment,
    SamplingHostStatusManifest,
    SamplingManifest,
    read_materialized_batches_manifest,
    read_phase_metrics_manifest,
    read_policy_manifest,
    read_run_state,
    read_sampling_host_status,
    read_sampling_manifest,
    sampling_host_statuses_exist,
    write_run_state,
    write_sampling_manifest,
)
from marin.rl.alternating.wandb import init_alternating_controller_tracker

logger = logging.getLogger(__name__)


class AlternatingPhaseHooks(Protocol):
    """Integration surface implemented by the phase-specific runtime."""

    def bootstrap_initial_policy(
        self,
        config: AlternatingRLConfig,
        paths: AlternatingRunPaths,
    ) -> tuple[str, int]:
        """Create policy `0000` and return its manifest path plus discovered host count."""
        ...

    def initialize_curriculum_state(self, config: AlternatingRLConfig, paths: AlternatingRunPaths) -> None:
        """Create the initial curriculum state file for a new run."""
        ...

    def frozen_lesson_weights(
        self,
        config: AlternatingRLConfig,
        state: AlternatingRunState,
        paths: AlternatingRunPaths,
    ) -> dict[str, float]:
        """Return the frozen lesson weights for the next sampling phase."""
        ...

    def launch_sampling_phase(
        self,
        config: AlternatingRLConfig,
        state: AlternatingRunState,
        manifest: SamplingManifest,
        paths: AlternatingRunPaths,
    ) -> None:
        """Launch one sampling host runner per TPU host."""
        ...

    def wait_for_sampling_phase(
        self,
        config: AlternatingRLConfig,
        state: AlternatingRunState,
        manifest: SamplingManifest,
        paths: AlternatingRunPaths,
    ) -> None:
        """Block until every sampling host has finished."""
        ...

    def update_curriculum_state(
        self,
        config: AlternatingRLConfig,
        state: AlternatingRunState,
        manifest: SamplingManifest,
        paths: AlternatingRunPaths,
    ) -> None:
        """Aggregate phase rollout stats into the durable curriculum state."""
        ...

    def materialize_training_batches(
        self,
        config: AlternatingRLConfig,
        state: AlternatingRunState,
        manifest: SamplingManifest,
        paths: AlternatingRunPaths,
    ) -> str:
        """Materialize raw rollouts and return the materialization manifest path."""
        ...

    def run_training_phase(
        self,
        config: AlternatingRLConfig,
        state: AlternatingRunState,
        manifest: MaterializedBatchesManifest,
        paths: AlternatingRunPaths,
    ) -> str:
        """Run one training phase and return the next policy manifest path."""
        ...


def ensure_image_digest_matches(config: AlternatingRLConfig, state: AlternatingRunState) -> None:
    """Fail fast if the requested image digest drifted mid-run."""
    if state.image_digest != config.image_digest:
        raise ValueError(
            "alternating RL image digest drifted: "
            f"run_state has {state.image_digest}, config requested {config.image_digest}"
        )


def build_sampling_host_assignments(
    config: AlternatingRLConfig,
    *,
    phase_id: int,
    num_hosts: int,
) -> list[SamplingHostAssignment]:
    """Derive one host assignment per TPU host for the next sampling phase."""
    train_group_targets = config.quotas.train_group_targets_by_host(num_hosts=num_hosts)
    phase_seed_base = config.seed + phase_id * num_hosts

    return [
        SamplingHostAssignment(
            host_ordinal=host_ordinal,
            seed=phase_seed_base + host_ordinal,
            target_train_groups=train_group_targets[host_ordinal],
        )
        for host_ordinal in range(num_hosts)
    ]


def bootstrap_or_resume(
    config: AlternatingRLConfig,
    hooks: AlternatingPhaseHooks,
    paths: AlternatingRunPaths,
) -> AlternatingRunState:
    """Create the initial run state or resume an existing one."""
    config.validate()

    if exists(paths.run_state_path):
        state = read_run_state(paths.run_state_path)
        ensure_image_digest_matches(config, state)
        return state

    policy_manifest_path, discovered_num_hosts = hooks.bootstrap_initial_policy(config, paths)
    if not exists(policy_manifest_path):
        raise FileNotFoundError(f"bootstrap policy manifest was not written: {policy_manifest_path}")
    if discovered_num_hosts != config.cluster.num_hosts:
        raise ValueError(
            "configured num_hosts does not match discovered TPU workers: "
            f"{config.cluster.num_hosts} != {discovered_num_hosts}"
        )

    hooks.initialize_curriculum_state(config, paths)
    if not exists(paths.curriculum_state_path):
        raise FileNotFoundError(f"initial curriculum state was not written: {paths.curriculum_state_path}")

    policy_manifest = read_policy_manifest(policy_manifest_path)
    state = AlternatingRunState(
        run_id=config.run_id,
        status=RunStatus.SAMPLING,
        phase_id=0,
        policy_version=policy_manifest.policy_version,
        source_global_step=policy_manifest.source_global_step,
        num_hosts=discovered_num_hosts,
        tpu_name=config.cluster.tpu_name,
        tpu_type=config.cluster.tpu_type,
        zone=config.cluster.zone,
        image_digest=config.image_digest,
        current_policy_manifest_path=policy_manifest_path,
        current_levanter_checkpoint_path=policy_manifest.levanter_checkpoint_path,
        current_sampling_manifest=None,
        current_materialized_manifest=None,
        last_completed_phase=-1,
    )
    write_run_state(paths.run_state_path, state)
    return state


def should_stop(state: AlternatingRunState, config: AlternatingRLConfig) -> bool:
    """Return whether the controller should stop before launching another phase."""
    if state.source_global_step >= config.quotas.num_train_steps:
        return True
    if config.quotas.max_phases is None:
        return False
    return state.last_completed_phase + 1 >= config.quotas.max_phases


def mark_completed(state: AlternatingRunState) -> AlternatingRunState:
    """Return the completed terminal state."""
    return replace(
        state,
        status=RunStatus.COMPLETED,
        current_sampling_manifest=None,
        current_materialized_manifest=None,
    )


def mark_failed(state: AlternatingRunState) -> AlternatingRunState:
    """Return the failed terminal state."""
    return replace(state, status=RunStatus.FAILED)


def advance_run_state(
    state: AlternatingRunState,
    next_policy_manifest_path: str,
) -> AlternatingRunState:
    """Advance durable state after one training/export phase succeeds."""
    if not exists(next_policy_manifest_path):
        raise FileNotFoundError(f"missing next policy manifest: {next_policy_manifest_path}")
    next_policy = read_policy_manifest(next_policy_manifest_path)
    if next_policy.policy_version <= state.policy_version:
        raise ValueError(
            "next policy did not advance policy_version: "
            f"current={state.policy_version}, next={next_policy.policy_version}"
        )
    if next_policy.phase_id != state.phase_id:
        raise ValueError(
            "next policy manifest phase_id does not match the completed phase: "
            f"state.phase_id={state.phase_id}, next_policy.phase_id={next_policy.phase_id}"
        )
    if next_policy.source_global_step < state.source_global_step:
        raise ValueError(
            "next policy source_global_step moved backwards: "
            f"current={state.source_global_step}, next={next_policy.source_global_step}"
        )

    return replace(
        state,
        status=RunStatus.SAMPLING,
        phase_id=state.phase_id + 1,
        policy_version=next_policy.policy_version,
        source_global_step=next_policy.source_global_step,
        current_policy_manifest_path=next_policy_manifest_path,
        current_levanter_checkpoint_path=next_policy.levanter_checkpoint_path,
        current_sampling_manifest=None,
        current_materialized_manifest=None,
        last_completed_phase=state.phase_id,
    )


def ensure_sampling_manifest(
    config: AlternatingRLConfig,
    state: AlternatingRunState,
    hooks: AlternatingPhaseHooks,
    paths: AlternatingRunPaths,
) -> tuple[AlternatingRunState, SamplingManifest]:
    """Load or create the current sampling manifest."""
    manifest_path = state.current_sampling_manifest
    default_manifest_path = paths.sampling_manifest_path(state.phase_id)
    if manifest_path is not None and exists(manifest_path):
        manifest = read_sampling_manifest(manifest_path)
    elif exists(default_manifest_path):
        manifest_path = default_manifest_path
        manifest = read_sampling_manifest(manifest_path)
    else:
        manifest_path = default_manifest_path
        manifest = SamplingManifest(
            phase_id=state.phase_id,
            policy_version=state.policy_version,
            policy_manifest_path=state.current_policy_manifest_path,
            curriculum_state_path=paths.curriculum_state_path,
            curriculum_snapshot_path=paths.sampling_curriculum_snapshot_path(state.phase_id),
            num_hosts=state.num_hosts,
            local_tensor_parallel_size=config.cluster.local_tensor_parallel_size,
            coordinator_host_ordinal=0,
            host_assignments=build_sampling_host_assignments(
                config,
                phase_id=state.phase_id,
                num_hosts=state.num_hosts,
            ),
            frozen_lesson_weights=hooks.frozen_lesson_weights(config, state, paths),
            rollout_output_root=paths.sampling_phase_dir(state.phase_id),
        )
        write_sampling_manifest(manifest_path, manifest)

    next_state = replace(
        state,
        status=RunStatus.SAMPLING,
        current_sampling_manifest=manifest_path,
        current_materialized_manifest=None,
    )
    write_run_state(paths.run_state_path, next_state)
    return next_state, manifest


def ensure_materialized_manifest(
    config: AlternatingRLConfig,
    state: AlternatingRunState,
    hooks: AlternatingPhaseHooks,
    sampling_manifest: SamplingManifest,
    paths: AlternatingRunPaths,
) -> tuple[AlternatingRunState, MaterializedBatchesManifest]:
    """Load or create the current materialized-batch manifest."""
    manifest_path = state.current_materialized_manifest
    default_manifest_path = paths.materialized_manifest_path(state.phase_id)
    if manifest_path is not None and exists(manifest_path):
        manifest = read_materialized_batches_manifest(manifest_path)
    elif exists(default_manifest_path):
        manifest_path = default_manifest_path
        manifest = read_materialized_batches_manifest(manifest_path)
    else:
        manifest_path = hooks.materialize_training_batches(config, state, sampling_manifest, paths)
        if not exists(manifest_path):
            raise FileNotFoundError(f"materialization hook did not write manifest: {manifest_path}")
        manifest = read_materialized_batches_manifest(manifest_path)

    next_state = replace(
        state,
        status=RunStatus.TRAINING,
        current_materialized_manifest=manifest_path,
    )
    write_run_state(paths.run_state_path, next_state)
    return next_state, manifest


def verify_sampling_host_statuses(
    manifest: SamplingManifest,
    paths: AlternatingRunPaths,
) -> list[SamplingHostStatusManifest]:
    """Read every host status file and ensure the phase succeeded."""
    if not sampling_host_statuses_exist(paths, manifest):
        raise FileNotFoundError(f"sampling phase completed without all host status files: phase_id={manifest.phase_id}")

    statuses: list[SamplingHostStatusManifest] = []
    for assignment in manifest.host_assignments:
        status_path = paths.sampling_host_status_path(manifest.phase_id, assignment.host_ordinal)
        status = read_sampling_host_status(status_path)
        statuses.append(status)
        if status.status != HostPhaseStatus.SUCCEEDED:
            raise RuntimeError(
                "sampling host reported failure: "
                f"phase_id={status.phase_id}, host_ordinal={status.host_ordinal}, error={status.error_message}"
            )

    return statuses


def _existing_sampling_status_count(paths: AlternatingRunPaths, manifest: SamplingManifest) -> int:
    return sum(
        int(exists(paths.sampling_host_status_path(manifest.phase_id, assignment.host_ordinal)))
        for assignment in manifest.host_assignments
    )


def _expected_next_policy_manifest_path(paths: AlternatingRunPaths, state: AlternatingRunState) -> str:
    return paths.policy_manifest_path(state.policy_version + 1)


def _phase_metrics_for_phase(paths: AlternatingRunPaths, phase_id: int):
    metrics_path = paths.phase_metrics_path(phase_id)
    if not exists(metrics_path):
        return None
    return read_phase_metrics_manifest(metrics_path)


def _log_phase_summary(paths: AlternatingRunPaths, phase_id: int):
    metrics = _phase_metrics_for_phase(paths, phase_id)
    if metrics is None:
        logger.info("alternating phase %d completed without a metrics manifest", phase_id)
        return None

    logger.info(
        "alternating phase %d timings: prepare_sampling=%s sampling=%s curriculum_update=%s "
        "materialization=%s training=%s export=%s total=%s",
        phase_id,
        metrics.prepare_sampling_seconds,
        metrics.sampling_seconds,
        metrics.curriculum_update_seconds,
        metrics.materialization_seconds,
        metrics.training_seconds,
        metrics.export_seconds,
        metrics.total_recorded_seconds,
    )
    return metrics


def _controller_hyperparameters(
    config: AlternatingRLConfig,
    *,
    resumed: bool,
) -> dict[str, object]:
    return {
        "alternating/run_id": config.run_id,
        "alternating/shared_root": config.shared_root,
        "alternating/image_digest": config.image_digest,
        "alternating/seed": config.seed,
        "alternating/resumed": resumed,
        "alternating/cluster/tpu_name": config.cluster.tpu_name,
        "alternating/cluster/tpu_type": config.cluster.tpu_type,
        "alternating/cluster/zone": config.cluster.zone,
        "alternating/cluster/num_hosts": config.cluster.num_hosts,
        "alternating/cluster/local_tensor_parallel_size": config.cluster.local_tensor_parallel_size,
        "alternating/cluster/capacity_type": config.cluster.capacity_type,
        "alternating/quotas/steps_per_phase": config.quotas.steps_per_phase,
        "alternating/quotas/num_train_steps": config.quotas.num_train_steps,
        "alternating/quotas/groups_per_training_step": config.quotas.groups_per_training_step,
        "alternating/quotas/eval_examples_per_lesson": config.quotas.eval_examples_per_lesson,
        "alternating/initial_checkpoint": config.initial_checkpoint,
        "alternating/tokenizer_name": config.tokenizer_name,
        "alternating/model_name": getattr(config.inference, "model_name", None),
    }


def _controller_metrics(
    state: AlternatingRunState,
    *,
    event: str,
    sampling_manifest: SamplingManifest | None = None,
    sampling_statuses: list[SamplingHostStatusManifest] | None = None,
    materialized_manifest: MaterializedBatchesManifest | None = None,
    phase_metrics=None,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "alternating/controller/event": event,
        "alternating/controller/status": state.status.value,
        "alternating/controller/phase_id": state.phase_id,
        "alternating/controller/policy_version": state.policy_version,
        "alternating/controller/source_global_step": state.source_global_step,
        "alternating/controller/last_completed_phase": state.last_completed_phase,
        "alternating/controller/num_hosts": state.num_hosts,
    }
    if sampling_manifest is not None:
        metrics["alternating/controller/expected_sampling_hosts"] = len(sampling_manifest.host_assignments)
        metrics["alternating/controller/local_tensor_parallel_size"] = sampling_manifest.local_tensor_parallel_size
        metrics["alternating/controller/frozen_lesson_count"] = len(sampling_manifest.frozen_lesson_weights)
        metrics["alternating/controller/target_train_groups"] = sum(
            assignment.target_train_groups for assignment in sampling_manifest.host_assignments
        )
    if sampling_statuses is not None:
        metrics["alternating/controller/sampling_hosts_succeeded"] = sum(
            int(status.status == HostPhaseStatus.SUCCEEDED) for status in sampling_statuses
        )
        metrics["alternating/controller/sampling_hosts_failed"] = sum(
            int(status.status == HostPhaseStatus.FAILED) for status in sampling_statuses
        )
        metrics["alternating/controller/sampling_rollout_files"] = sum(
            len(status.rollout_file_paths) for status in sampling_statuses
        )
        metrics["alternating/controller/sampling_train_groups"] = sum(
            status.num_train_groups for status in sampling_statuses
        )
    if materialized_manifest is not None:
        metrics["alternating/controller/materialized_rollout_groups"] = materialized_manifest.num_rollout_groups
        metrics["alternating/controller/materialized_rollouts"] = materialized_manifest.num_individual_rollouts
        metrics["alternating/controller/materialized_training_batches"] = materialized_manifest.num_training_batches
        metrics["alternating/controller/materialized_global_batch_size"] = materialized_manifest.global_batch_size
    if phase_metrics is not None:
        metrics["alternating/phase_total_seconds"] = phase_metrics.total_recorded_seconds
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

    return metrics


def _log_controller_event(
    tracker,
    state: AlternatingRunState,
    *,
    event: str,
    sampling_manifest: SamplingManifest | None = None,
    sampling_statuses: list[SamplingHostStatusManifest] | None = None,
    materialized_manifest: MaterializedBatchesManifest | None = None,
    phase_metrics=None,
) -> None:
    if tracker is None:
        return

    tracker.log(
        _controller_metrics(
            state,
            event=event,
            sampling_manifest=sampling_manifest,
            sampling_statuses=sampling_statuses,
            materialized_manifest=materialized_manifest,
            phase_metrics=phase_metrics,
        ),
        step=state.source_global_step,
    )


def _log_controller_summary(
    tracker,
    state: AlternatingRunState,
    *,
    error_message: str | None = None,
) -> None:
    if tracker is None:
        return

    summary = {
        "alternating/controller/final_status": state.status.value,
        "alternating/controller/final_phase_id": state.phase_id,
        "alternating/controller/final_policy_version": state.policy_version,
        "alternating/controller/final_source_global_step": state.source_global_step,
        "alternating/controller/last_completed_phase": state.last_completed_phase,
    }
    if error_message is not None:
        summary["alternating/controller/error_message"] = error_message
    tracker.log_summary(summary)


def run_controller(
    config: AlternatingRLConfig,
    hooks: AlternatingPhaseHooks,
) -> AlternatingRunState:
    """Run the explicit alternating RL controller loop."""
    paths = AlternatingRunPaths.from_config(config)
    resumed = exists(paths.run_state_path)
    state = bootstrap_or_resume(config, hooks, paths)
    tracker = init_alternating_controller_tracker(config, paths)
    tracker_finished = False
    if tracker is not None:
        tracker.log_hyperparameters(_controller_hyperparameters(config, resumed=resumed))
        _log_controller_event(
            tracker,
            state,
            event="controller_resumed" if resumed else "controller_started",
        )

    try:
        while True:
            ensure_image_digest_matches(config, state)

            if state.status == RunStatus.COMPLETED:
                _log_controller_event(tracker, state, event="completed")
                _log_controller_summary(tracker, state)
                return state
            if state.status == RunStatus.FAILED:
                _log_controller_event(tracker, state, event="failed")
                _log_controller_summary(tracker, state)
                return state
            if should_stop(state, config):
                completed_state = mark_completed(state)
                write_run_state(paths.run_state_path, completed_state)
                _log_controller_event(tracker, completed_state, event="completed")
                _log_controller_summary(tracker, completed_state)
                return completed_state

            if state.status == RunStatus.SAMPLING:
                state, sampling_manifest = ensure_sampling_manifest(config, state, hooks, paths)
                _log_controller_event(
                    tracker,
                    state,
                    event="sampling_manifest_ready",
                    sampling_manifest=sampling_manifest,
                )
                sampling_status_count = _existing_sampling_status_count(paths, sampling_manifest)
                if sampling_status_count == 0:
                    hooks.launch_sampling_phase(config, state, sampling_manifest, paths)
                    hooks.wait_for_sampling_phase(config, state, sampling_manifest, paths)
                elif sampling_status_count < len(sampling_manifest.host_assignments):
                    hooks.wait_for_sampling_phase(config, state, sampling_manifest, paths)
                sampling_statuses = verify_sampling_host_statuses(sampling_manifest, paths)
                hooks.update_curriculum_state(config, state, sampling_manifest, paths)
                _log_controller_event(
                    tracker,
                    state,
                    event="sampling_completed",
                    sampling_manifest=sampling_manifest,
                    sampling_statuses=sampling_statuses,
                    phase_metrics=_phase_metrics_for_phase(paths, state.phase_id),
                )
                state = replace(state, status=RunStatus.MATERIALIZING)
                write_run_state(paths.run_state_path, state)
                _log_controller_event(tracker, state, event="materialization_started")
                continue

            if state.status == RunStatus.MATERIALIZING:
                if state.current_sampling_manifest is None:
                    raise ValueError("state.current_sampling_manifest must be set during materialization")
                sampling_manifest = read_sampling_manifest(state.current_sampling_manifest)
                state, materialized_manifest = ensure_materialized_manifest(
                    config, state, hooks, sampling_manifest, paths
                )
                _log_controller_event(
                    tracker,
                    state,
                    event="materialization_completed",
                    materialized_manifest=materialized_manifest,
                    phase_metrics=_phase_metrics_for_phase(paths, state.phase_id),
                )
                continue

            if state.status == RunStatus.TRAINING:
                if state.current_materialized_manifest is None:
                    raise ValueError("state.current_materialized_manifest must be set during training")
                materialized_manifest = read_materialized_batches_manifest(state.current_materialized_manifest)
                next_policy_manifest_path = _expected_next_policy_manifest_path(paths, state)
                if not exists(next_policy_manifest_path):
                    next_policy_manifest_path = hooks.run_training_phase(config, state, materialized_manifest, paths)
                phase_metrics = _log_phase_summary(paths, state.phase_id)
                _log_controller_event(
                    tracker,
                    state,
                    event="training_completed",
                    materialized_manifest=materialized_manifest,
                    phase_metrics=phase_metrics,
                )
                state = advance_run_state(state, next_policy_manifest_path)
                write_run_state(paths.run_state_path, state)
                _log_controller_event(tracker, state, event="phase_advanced")
                continue

            raise AssertionError(f"Unexpected controller state: {state.status}")
    except Exception:
        failed_state = mark_failed(state)
        write_run_state(paths.run_state_path, failed_state)
        _log_controller_event(
            tracker,
            failed_state,
            event="failed",
            phase_metrics=_phase_metrics_for_phase(paths, failed_state.phase_id),
        )
        _log_controller_summary(tracker, failed_state, error_message=traceback.format_exc())
        raise
    finally:
        if tracker is not None and not tracker_finished:
            tracker.finish()
            tracker_finished = True
