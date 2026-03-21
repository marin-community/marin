# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller loop for alternating single-pod RL."""

from __future__ import annotations

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


def _log_phase_summary(paths: AlternatingRunPaths, phase_id: int) -> None:
    metrics_path = paths.phase_metrics_path(phase_id)
    if not exists(metrics_path):
        logger.info("alternating phase %d completed without a metrics manifest", phase_id)
        return

    metrics = read_phase_metrics_manifest(metrics_path)
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


def run_controller(
    config: AlternatingRLConfig,
    hooks: AlternatingPhaseHooks,
) -> AlternatingRunState:
    """Run the explicit alternating RL controller loop."""
    paths = AlternatingRunPaths.from_config(config)
    state = bootstrap_or_resume(config, hooks, paths)

    try:
        while True:
            ensure_image_digest_matches(config, state)

            if state.status == RunStatus.COMPLETED:
                return state
            if state.status == RunStatus.FAILED:
                return state
            if should_stop(state, config):
                completed_state = mark_completed(state)
                write_run_state(paths.run_state_path, completed_state)
                return completed_state

            if state.status == RunStatus.SAMPLING:
                state, sampling_manifest = ensure_sampling_manifest(config, state, hooks, paths)
                sampling_status_count = _existing_sampling_status_count(paths, sampling_manifest)
                if sampling_status_count == 0:
                    hooks.launch_sampling_phase(config, state, sampling_manifest, paths)
                    hooks.wait_for_sampling_phase(config, state, sampling_manifest, paths)
                elif sampling_status_count < len(sampling_manifest.host_assignments):
                    hooks.wait_for_sampling_phase(config, state, sampling_manifest, paths)
                verify_sampling_host_statuses(sampling_manifest, paths)
                hooks.update_curriculum_state(config, state, sampling_manifest, paths)
                state = replace(state, status=RunStatus.MATERIALIZING)
                write_run_state(paths.run_state_path, state)
                continue

            if state.status == RunStatus.MATERIALIZING:
                if state.current_sampling_manifest is None:
                    raise ValueError("state.current_sampling_manifest must be set during materialization")
                sampling_manifest = read_sampling_manifest(state.current_sampling_manifest)
                state, _ = ensure_materialized_manifest(config, state, hooks, sampling_manifest, paths)
                continue

            if state.status == RunStatus.TRAINING:
                if state.current_materialized_manifest is None:
                    raise ValueError("state.current_materialized_manifest must be set during training")
                materialized_manifest = read_materialized_batches_manifest(state.current_materialized_manifest)
                next_policy_manifest_path = _expected_next_policy_manifest_path(paths, state)
                if not exists(next_policy_manifest_path):
                    next_policy_manifest_path = hooks.run_training_phase(config, state, materialized_manifest, paths)
                _log_phase_summary(paths, state.phase_id)
                state = advance_run_state(state, next_policy_manifest_path)
                write_run_state(paths.run_state_path, state)
                continue

            raise AssertionError(f"Unexpected controller state: {state.status}")
    except Exception:
        failed_state = mark_failed(state)
        write_run_state(paths.run_state_path, failed_state)
        raise
