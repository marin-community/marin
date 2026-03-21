# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase orchestration controller for alternating RL.

The controller runs on a local machine or CPU VM. It:
- creates/reuses the TPU pod
- manages run state
- launches sampling on each host, waits for completion
- runs the materializer on worker 0
- launches full-pod training
- exports the next policy
- handles resume from GCS state
"""

import logging
import os
import time
from datetime import datetime, timezone

from levanter.infra.tpus import (
    ensure_tpu_exists,
    resolve_image_digest,
    run_container_all_workers,
    run_container_on_worker,
    stop_container_all_workers,
    stop_container_on_worker,
    worker_health_probe,
)

from marin.rl.alternating.config import AlternatingRLConfig
from marin.rl.alternating.local_topology import local_vllm_topology, num_hosts_for_tpu_type
from marin.rl.alternating.state import (
    AlternatingRunState,
    HostStatus,
    MaterializationManifest,
    PolicyManifest,
    RunStatus,
    SamplingHostAssignment,
    SamplingManifest,
    path_exists,
    read_json_from_path,
    write_json_to_path,
)
from marin.rl.curriculum import Curriculum

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 15


def run_controller(config: AlternatingRLConfig) -> None:
    """Main controller loop for alternating RL."""
    num_hosts = num_hosts_for_tpu_type(config.tpu_type)
    logger.info(
        "Starting alternating RL controller: run_id=%s, tpu=%s (%d hosts)", config.run_id, config.tpu_type, num_hosts
    )

    # Ensure TPU and image exist
    image_digest = _ensure_image_and_tpu(config)
    state = _bootstrap_or_resume(config, num_hosts, image_digest)

    while not _should_stop(state, config):
        logger.info(
            "=== Phase %d (policy_version=%d, status=%s) ===", state.phase_id, state.policy_version, state.status
        )

        # --- Sampling ---
        if state.status == RunStatus.SAMPLING:
            sampling_manifest = _write_sampling_manifest(config, state, num_hosts)
            state.current_sampling_manifest = config.sampling_manifest_path(state.phase_id)
            _update_run_state(config, state)

            _launch_sampling_phase(config, state, sampling_manifest, num_hosts)
            _wait_for_sampling_hosts(config, state, sampling_manifest, num_hosts)

            # Aggregate eval stats and update curriculum
            _aggregate_eval_and_update_curriculum(config, state)

            state.status = RunStatus.MATERIALIZING
            _update_run_state(config, state)

        # --- Materialization ---
        if state.status == RunStatus.MATERIALIZING:
            _launch_materialization(config, state, num_hosts)
            _wait_for_materialization(config, state)

            state.current_materialized_manifest = config.materialized_manifest_path(state.phase_id)
            state.status = RunStatus.TRAINING
            _update_run_state(config, state)

        # --- Training ---
        if state.status == RunStatus.TRAINING:
            _launch_training_phase(config, state, num_hosts)
            _wait_for_training(config, state)

            state.status = RunStatus.EXPORTING
            _update_run_state(config, state)

        # --- Export ---
        if state.status == RunStatus.EXPORTING:
            # Export happens as part of the training phase process.
            # If we get here with status=exporting, training completed but
            # we need to verify the policy manifest exists.
            next_version = state.policy_version + 1
            next_policy_manifest_path = config.policy_manifest_path(next_version)

            if not path_exists(next_policy_manifest_path):
                logger.error(
                    "Training completed but policy manifest %s not found. "
                    "The training phase should have exported it.",
                    next_policy_manifest_path,
                )
                state.status = RunStatus.FAILED
                _update_run_state(config, state)
                raise RuntimeError(f"Policy manifest missing: {next_policy_manifest_path}")

            # Read the new policy manifest to get the checkpoint path
            new_policy = PolicyManifest.from_json(read_json_from_path(next_policy_manifest_path))

            # Advance state
            state.policy_version = next_version
            state.source_global_step = new_policy.source_global_step
            state.current_policy_path = new_policy.hf_export_path
            state.current_levanter_checkpoint_path = new_policy.levanter_checkpoint_path
            state.last_completed_phase = state.phase_id
            state.phase_id += 1
            state.status = RunStatus.SAMPLING
            state.current_sampling_manifest = None
            state.current_materialized_manifest = None
            _update_run_state(config, state)
            logger.info(
                "Phase %d completed. Advancing to phase %d with policy_version=%d",
                state.last_completed_phase,
                state.phase_id,
                state.policy_version,
            )

    logger.info(
        "Alternating RL run %s completed at phase %d, step %d", config.run_id, state.phase_id, state.source_global_step
    )
    state.status = RunStatus.COMPLETED
    _update_run_state(config, state)


# ---------------------------------------------------------------------------
# Bootstrap / Resume
# ---------------------------------------------------------------------------


def _ensure_image_and_tpu(config: AlternatingRLConfig) -> str:
    """Ensure the TPU exists and resolve the image digest."""
    ensure_tpu_exists(
        tpu_name=config.tpu_name,
        tpu_type=config.tpu_type,
        zone=config.zone,
        capacity_type=config.capacity_type,
    )

    if config.image:
        image_digest = resolve_image_digest(config.image)
    else:
        # In production this would auto-build; for now require explicit image
        raise ValueError("config.image must be set to a Docker image tag or digest")

    logger.info("Image digest: %s", image_digest)
    return image_digest


def _bootstrap_or_resume(config: AlternatingRLConfig, num_hosts: int, image_digest: str) -> AlternatingRunState:
    """Load existing run state or create a new one."""
    if path_exists(config.run_state_path):
        state = AlternatingRunState.from_json(read_json_from_path(config.run_state_path))
        logger.info("Resuming run %s at phase %d, status=%s", state.run_id, state.phase_id, state.status)

        # Verify image digest matches
        if state.image_digest != image_digest:
            raise RuntimeError(
                f"Image digest mismatch: run_state has {state.image_digest}, "
                f"but current image resolves to {image_digest}. "
                "Mid-run image drift is not allowed."
            )

        return state

    # Fresh run: bootstrap
    logger.info("Bootstrapping new run %s", config.run_id)

    # Initialize curriculum
    curriculum = Curriculum(config.curriculum)
    curriculum.save_checkpoint(config.curriculum_dir)

    # Prepare initial policy
    # If initial_checkpoint is an HF model, the policy_0000 directory is just
    # the HF model itself. The sampling host will use fast bootstrap from it.
    # Write a policy manifest for the initial checkpoint
    policy_manifest = PolicyManifest(
        policy_version=0,
        phase_id=-1,
        source_global_step=0,
        hf_export_path=config.initial_checkpoint,
        levanter_checkpoint_path=None,
        model_name=config.model_name_or_path,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_to_path(config.policy_manifest_path(0), policy_manifest.to_json())

    state = AlternatingRunState(
        run_id=config.run_id,
        status=RunStatus.SAMPLING,
        phase_id=0,
        policy_version=0,
        source_global_step=0,
        num_hosts=num_hosts,
        tpu_name=config.tpu_name,
        tpu_type=config.tpu_type,
        zone=config.zone,
        image_digest=image_digest,
        current_policy_path=config.initial_checkpoint,
        current_levanter_checkpoint_path=None,
    )
    _update_run_state(config, state)
    return state


def _should_stop(state: AlternatingRunState, config: AlternatingRLConfig) -> bool:
    """Check whether the run should stop."""
    if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
        return True
    if config.max_phases is not None and state.phase_id >= config.max_phases:
        return True
    if state.source_global_step >= config.trainer.num_train_steps:
        return True
    return False


def _update_run_state(config: AlternatingRLConfig, state: AlternatingRunState) -> None:
    write_json_to_path(config.run_state_path, state.to_json())


# ---------------------------------------------------------------------------
# Sampling Phase
# ---------------------------------------------------------------------------


def _write_sampling_manifest(
    config: AlternatingRLConfig,
    state: AlternatingRunState,
    num_hosts: int,
) -> SamplingManifest:
    """Compute frozen curriculum weights and write the sampling manifest."""
    curriculum = Curriculum(config.curriculum)
    curriculum.restore_checkpoint(config.curriculum_dir)

    frozen_weights = curriculum.compute_sampling_weights()
    if not frozen_weights:
        # No active lessons - use uniform over all lessons
        lesson_ids = list(config.curriculum.lessons.keys())
        frozen_weights = {lid: 1.0 / len(lesson_ids) for lid in lesson_ids}

    topology = local_vllm_topology(config.tpu_type)
    train_groups_per_host = config.required_train_groups_per_host(num_hosts)

    assignments = []
    for i in range(num_hosts):
        assignments.append(
            SamplingHostAssignment(
                host_ordinal=i,
                seed=config.seed + state.phase_id * 1000 + i,
                target_train_groups=train_groups_per_host,
                target_eval_groups=config.eval_groups_per_host,
            )
        )

    manifest = SamplingManifest(
        phase_id=state.phase_id,
        policy_version=state.policy_version,
        policy_manifest_path=config.policy_manifest_path(state.policy_version),
        curriculum_state_path=os.path.join(config.curriculum_dir, "curriculum_state.json"),
        num_hosts=num_hosts,
        local_tensor_parallel_size=topology.tensor_parallel_size,
        host_assignments=assignments,
        frozen_lesson_weights=frozen_weights,
        output_root=config.sampling_phase_dir(state.phase_id),
    )

    manifest_path = config.sampling_manifest_path(state.phase_id)
    write_json_to_path(manifest_path, manifest.to_json())
    logger.info("Sampling manifest written to %s", manifest_path)
    return manifest


def _launch_sampling_phase(
    config: AlternatingRLConfig,
    state: AlternatingRunState,
    manifest: SamplingManifest,
    num_hosts: int,
) -> None:
    """Launch sampling containers on each host."""
    topology = local_vllm_topology(config.tpu_type)

    for i in range(num_hosts):
        # Clean up any existing sampler container
        stop_container_on_worker(
            config.tpu_name,
            config.zone,
            i,
            name=config.sampler_container_name,
        )

        # Health probe
        if not worker_health_probe(config.tpu_name, config.zone, i):
            raise RuntimeError(f"Worker {i} failed health probe before sampling")

        # Build sampling command
        cmd = [
            "uv",
            "run",
            "python",
            "experiments/alternating_rl_math500.py",
            "--mode",
            "sampling-host",
            "--manifest-path",
            config.sampling_manifest_path(state.phase_id),
            "--host-ordinal",
            str(i),
        ]

        # Build env vars
        env = {**topology.env_vars(), **config.extra_env}
        env["JAX_COMPILATION_CACHE_DIR"] = config.sampling_compilation_cache_dir

        run_container_on_worker(
            tpu_name=config.tpu_name,
            zone=config.zone,
            worker_ordinal=i,
            image_id=state.image_digest,
            command=cmd,
            env=env,
            name=config.sampler_container_name,
            foreground=False,
        )
        logger.info("Launched sampler on worker %d", i)


def _wait_for_sampling_hosts(
    config: AlternatingRLConfig,
    state: AlternatingRunState,
    manifest: SamplingManifest,
    num_hosts: int,
) -> None:
    """Poll for all host status.json files."""
    logger.info("Waiting for %d sampling hosts to complete...", num_hosts)
    while True:
        all_done = True
        for i in range(num_hosts):
            status_path = config.host_status_path(state.phase_id, i)
            if not path_exists(status_path):
                all_done = False
                break

        if all_done:
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    # Verify all succeeded
    for i in range(num_hosts):
        status_path = config.host_status_path(state.phase_id, i)
        host_status = HostStatus.from_json(read_json_from_path(status_path))
        if not host_status.success:
            raise RuntimeError(f"Sampling host {i} failed: {host_status.error}")

    logger.info("All %d sampling hosts completed successfully", num_hosts)

    # Stop sampler containers
    stop_container_all_workers(
        config.tpu_name,
        config.zone,
        num_hosts,
        name=config.sampler_container_name,
    )


# ---------------------------------------------------------------------------
# Curriculum update
# ---------------------------------------------------------------------------


def _aggregate_eval_and_update_curriculum(
    config: AlternatingRLConfig,
    state: AlternatingRunState,
) -> None:
    """Read eval rollouts from sampling phase and update curriculum state."""
    import pickle

    from iris.marin_fs import url_to_fs
    from marin.rl.types import RolloutBatch, RolloutStats

    curriculum = Curriculum(config.curriculum)
    curriculum.restore_checkpoint(config.curriculum_dir)

    # Read eval rollouts from all hosts
    eval_stats: list[RolloutStats] = []
    for ha in range(state.num_hosts):
        eval_dir = f"{config.sampling_phase_dir(state.phase_id)}/host_{ha:03d}/eval"
        fs, _ = url_to_fs(eval_dir)
        if not fs.exists(eval_dir):
            continue
        for entry in sorted(fs.ls(eval_dir)):
            if isinstance(entry, dict):
                entry = entry["name"]
            if not entry.endswith(".pkl"):
                continue
            with fs.open(entry, "rb") as f:
                batch: RolloutBatch = pickle.load(f)
            for group in batch.groups:
                for rollout in group.rollouts:
                    eval_stats.append(
                        RolloutStats(
                            episode_reward=rollout.episode_reward,
                            env_example_id=rollout.env_example_id,
                            lesson_id=rollout.env_name,
                            temperature=rollout.temperature,
                            top_k=rollout.top_k if rollout.top_k is not None else -1,
                        )
                    )

    if eval_stats:
        curriculum.update_lesson_stats(eval_stats, mode="eval", current_step=state.source_global_step)
        logger.info("Updated curriculum with %d eval stats from phase %d", len(eval_stats), state.phase_id)

    curriculum.save_checkpoint(config.curriculum_dir)


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def _launch_materialization(
    config: AlternatingRLConfig,
    state: AlternatingRunState,
    num_hosts: int,
) -> None:
    """Launch the materializer on worker 0."""
    stop_container_on_worker(
        config.tpu_name,
        config.zone,
        0,
        name=config.materializer_container_name,
    )

    if not worker_health_probe(config.tpu_name, config.zone, 0):
        raise RuntimeError("Worker 0 failed health probe before materialization")

    cmd = [
        "uv",
        "run",
        "python",
        "experiments/alternating_rl_math500.py",
        "--mode",
        "materialize",
        "--manifest-path",
        config.sampling_manifest_path(state.phase_id),
        "--output-dir",
        config.materialized_phase_dir(state.phase_id),
    ]

    env = {**config.extra_env}

    run_container_on_worker(
        tpu_name=config.tpu_name,
        zone=config.zone,
        worker_ordinal=0,
        image_id=state.image_digest,
        command=cmd,
        env=env,
        name=config.materializer_container_name,
        foreground=True,
    )


def _wait_for_materialization(
    config: AlternatingRLConfig,
    state: AlternatingRunState,
) -> None:
    """Wait for the materialization manifest to appear."""
    manifest_path = config.materialized_manifest_path(state.phase_id)
    logger.info("Waiting for materialization manifest at %s", manifest_path)
    while not path_exists(manifest_path):
        time.sleep(POLL_INTERVAL_SECONDS)

    mat_manifest = MaterializationManifest.from_json(read_json_from_path(manifest_path))
    logger.info(
        "Materialization complete: %d batches, %d groups, %d rollouts",
        mat_manifest.num_training_batches,
        mat_manifest.num_rollout_groups,
        mat_manifest.num_individual_rollouts,
    )

    # Stop materializer container
    stop_container_on_worker(
        config.tpu_name,
        config.zone,
        0,
        name=config.materializer_container_name,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _launch_training_phase(
    config: AlternatingRLConfig,
    state: AlternatingRunState,
    num_hosts: int,
) -> None:
    """Launch full-pod training across all workers."""
    stop_container_all_workers(
        config.tpu_name,
        config.zone,
        num_hosts,
        name=config.trainer_container_name,
    )

    # Health probe all workers
    for i in range(num_hosts):
        if not worker_health_probe(config.tpu_name, config.zone, i):
            raise RuntimeError(f"Worker {i} failed health probe before training")

    cmd = [
        "uv",
        "run",
        "python",
        "experiments/alternating_rl_math500.py",
        "--mode",
        "train-phase",
        "--manifest-path",
        config.materialized_manifest_path(state.phase_id),
        "--run-state-path",
        config.run_state_path,
    ]

    env = {
        "TPU_BACKEND_TYPE": "jax",
        "PJRT_DEVICE": "TPU",
        "JAX_COMPILATION_CACHE_DIR": config.training_compilation_cache_dir,
        **config.extra_env,
    }

    run_container_all_workers(
        tpu_name=config.tpu_name,
        zone=config.zone,
        num_workers=num_hosts,
        image_id=state.image_digest,
        command=cmd,
        env=env,
        name=config.trainer_container_name,
        foreground=True,
    )


def _wait_for_training(
    config: AlternatingRLConfig,
    state: AlternatingRunState,
) -> None:
    """Wait for training to complete by checking for the policy manifest.

    Training writes a policy manifest as its final step before exiting.
    """
    next_policy_version = state.policy_version + 1
    policy_manifest_path = config.policy_manifest_path(next_policy_version)
    logger.info("Waiting for training to complete (policy manifest at %s)", policy_manifest_path)

    while not path_exists(policy_manifest_path):
        time.sleep(POLL_INTERVAL_SECONDS)

    logger.info("Training phase complete, policy %d available", next_policy_version)

    # Stop training containers
    stop_container_all_workers(
        config.tpu_name,
        config.zone,
        state.num_hosts,
        name=config.trainer_container_name,
    )
