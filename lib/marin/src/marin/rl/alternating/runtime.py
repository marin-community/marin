# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Existing-pod phase hooks for alternating single-pod RL."""

from __future__ import annotations

import logging
import subprocess
import time

from levanter.infra import tpus
from levanter.utils.fsspec_utils import exists

from marin.rl.alternating.config import (
    MATERIALIZER_CONTAINER_NAME,
    SAMPLER_CONTAINER_NAME,
    TRAINER_CONTAINER_NAME,
    AlternatingRLConfig,
)
from marin.rl.alternating.controller import AlternatingPhaseHooks
from marin.rl.alternating.io import read_pickle, write_pickle
from marin.rl.alternating.materialization import run_materialization
from marin.rl.alternating.sampling_phase import prepare_sampling_phase, run_sampling_host
from marin.rl.alternating.state import (
    AlternatingRunPaths,
    AlternatingRunState,
    HostPhaseStatus,
    PhaseMetricsManifest,
    PolicyManifest,
    SamplingManifest,
    SamplingHostStatusManifest,
    read_sampling_host_status,
    sampling_host_statuses_exist,
    update_phase_metrics,
    utc_now_iso,
    write_policy_manifest,
)
from marin.rl.alternating.training_phase import export_policy_only
from marin.rl.alternating.training_phase import run_training_phase as run_training_phase_local
from marin.rl.curriculum import create_local_curriculum
from marin.rl.types import RolloutStats

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 30
DEFAULT_SAMPLING_TIMEOUT = 6 * 60 * 60
DEFAULT_MISSING_CONTAINER_GRACE_POLLS = 2


def resolve_container_image(image: str) -> str:
    """Resolve a stable image digest once at controller bootstrap."""
    if "@sha256:" in image:
        return image

    output = subprocess.check_output(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            image,
            "--format",
            "{{json .Manifest.Digest}}",
        ],
        text=True,
    ).strip()
    digest = output.strip('"')
    if not digest.startswith("sha256:"):
        raise ValueError(f"could not resolve sha256 digest for image {image}: {output}")
    last_segment = image.rsplit("/", 1)[-1]
    base = image.rsplit(":", 1)[0] if ":" in last_segment else image
    return f"{base}@{digest}"


def save_controller_config(config: AlternatingRLConfig, paths: AlternatingRunPaths) -> None:
    """Persist the full controller config for remote phase entrypoints."""
    write_pickle(paths.controller_config_path, config)


def _common_phase_env(config: AlternatingRLConfig, *, compilation_cache_dir: str) -> dict[str, str]:
    env = dict(config.env)
    env["JAX_COMPILATION_CACHE_DIR"] = compilation_cache_dir
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    if compilation_cache_dir.endswith("/vllm"):
        env["VLLM_XLA_CACHE_PATH"] = compilation_cache_dir
    return env


def _phase_command(
    paths: AlternatingRunPaths, subcommand: str, *, phase_id: int, host_ordinal: int | None = None
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "experiments/alternating_rl_math500.py",
        subcommand,
        "--config-path",
        paths.controller_config_path,
        "--phase-id",
        str(phase_id),
    ]
    if host_ordinal is not None:
        command.extend(["--host-ordinal", str(host_ordinal)])
    return command


def _build_training_rollout_stats(
    config: AlternatingRLConfig, lesson_rewards: dict[str, list[float]]
) -> list[RolloutStats]:
    stats: list[RolloutStats] = []
    for lesson_id, rewards in lesson_rewards.items():
        lesson = config.curriculum.lessons[lesson_id]
        for reward in rewards:
            stats.append(
                RolloutStats(
                    episode_reward=float(reward),
                    env_example_id="alternating",
                    lesson_id=lesson_id,
                    temperature=float(lesson.sampling_params.temperature),
                    top_k=lesson.sampling_params.top_k,
                )
            )
    return stats


def _sampling_phase_failure(statuses: list[SamplingHostStatusManifest]) -> SamplingHostStatusManifest | None:
    for status in statuses:
        if status.status == HostPhaseStatus.FAILED:
            return status
    return None


def _written_sampling_statuses(
    paths: AlternatingRunPaths,
    manifest: SamplingManifest,
) -> list[SamplingHostStatusManifest]:
    statuses: list[SamplingHostStatusManifest] = []
    for assignment in manifest.host_assignments:
        status_path = paths.sampling_host_status_path(manifest.phase_id, assignment.host_ordinal)
        if exists(status_path):
            statuses.append(read_sampling_host_status(status_path))
    return statuses


def _log_phase_metrics(
    paths: AlternatingRunPaths,
    phase_id: int,
    **updates: float | None,
) -> PhaseMetricsManifest:
    return update_phase_metrics(paths.phase_metrics_path(phase_id), phase_id=phase_id, **updates)


class ExistingPodPhaseHooks(AlternatingPhaseHooks):
    """Concrete pod/runtime hooks for the alternating RL controller."""

    def bootstrap_initial_policy(self, config: AlternatingRLConfig, paths: AlternatingRunPaths) -> tuple[str, int]:
        tpus.start_tpu_vm_queued_resources(
            tpu_name=config.cluster.tpu_name,
            tpu_type=config.cluster.tpu_type,
            capacity_type=config.cluster.capacity_type,
            version=config.cluster.runtime_version,
            zone=config.cluster.zone,
            node_count=config.cluster.node_count,
        )
        tpus.setup_vm_docker(
            config.cluster.tpu_name,
            config.cluster.zone,
            config.cluster.node_count,
        )
        tpus.probe_tpu_all_workers_health(
            config.cluster.tpu_name,
            config.cluster.zone,
            config.cluster.node_count,
        )
        workers = tpus.describe_tpu_workers(
            config.cluster.tpu_name,
            config.cluster.zone,
            config.cluster.node_count,
        )
        if len(workers) != config.cluster.num_hosts:
            raise ValueError(
                "configured num_hosts does not match discovered TPU workers: "
                f"{config.cluster.num_hosts} != {len(workers)}"
            )

        policy_source = config.initial_checkpoint or config.inference.model_name
        if policy_source is None:
            raise ValueError("alternating RL requires an initial checkpoint or inference model name")
        enable_fast_bootstrap = policy_source.startswith(("gs://", "s3://"))
        manifest = PolicyManifest(
            policy_version=0,
            phase_id=-1,
            source_global_step=0,
            policy_path=policy_source,
            levanter_checkpoint_path=None,
            model_name=config.inference.model_name,
            tokenizer_name=config.tokenizer_name,
            enable_fast_bootstrap=enable_fast_bootstrap,
            created_at=utc_now_iso(),
        )
        manifest_path = paths.policy_manifest_path(0)
        write_policy_manifest(manifest_path, manifest)
        return manifest_path, len(workers)

    def initialize_curriculum_state(self, config: AlternatingRLConfig, paths: AlternatingRunPaths) -> None:
        curriculum = create_local_curriculum(config.curriculum)
        curriculum.save_checkpoint(paths.curriculum_root)

    def frozen_lesson_weights(
        self,
        config: AlternatingRLConfig,
        state: AlternatingRunState,
        paths: AlternatingRunPaths,
    ) -> dict[str, float]:
        del state
        curriculum = create_local_curriculum(config.curriculum, checkpoint_path=paths.curriculum_root)
        return curriculum.compute_sampling_weights()

    def launch_sampling_phase(
        self,
        config: AlternatingRLConfig,
        state: AlternatingRunState,
        manifest: SamplingManifest,
        paths: AlternatingRunPaths,
    ) -> None:
        del state
        tpus.stop_container_all_workers(
            config.cluster.tpu_name,
            config.cluster.zone,
            config.cluster.node_count,
            name=SAMPLER_CONTAINER_NAME,
        )
        tpus.probe_tpu_all_workers_health(
            config.cluster.tpu_name,
            config.cluster.zone,
            config.cluster.node_count,
        )
        prepare_sampling_start = time.time()
        tpus.run_container_on_worker(
            tpu_name=config.cluster.tpu_name,
            zone=config.cluster.zone,
            node_count=config.cluster.node_count,
            worker_ordinal=manifest.coordinator_host_ordinal,
            full_image_id=config.image_digest,
            command=_phase_command(paths, "prepare-sampling", phase_id=manifest.phase_id),
            env=_common_phase_env(
                config,
                compilation_cache_dir=config.caches.sampling_compilation_cache_dir,
            ),
            foreground=True,
            name=SAMPLER_CONTAINER_NAME,
        )
        _log_phase_metrics(
            paths,
            manifest.phase_id,
            prepare_sampling_seconds=time.time() - prepare_sampling_start,
        )
        tpus.stop_container_on_worker(
            config.cluster.tpu_name,
            config.cluster.zone,
            config.cluster.node_count,
            manifest.coordinator_host_ordinal,
            name=SAMPLER_CONTAINER_NAME,
        )

        for assignment in manifest.host_assignments:
            env = _common_phase_env(
                config,
                compilation_cache_dir=config.caches.sampling_compilation_cache_dir,
            )
            env["ALT_RL_HOST_ORDINAL"] = str(assignment.host_ordinal)
            tpus.run_container_on_worker(
                tpu_name=config.cluster.tpu_name,
                zone=config.cluster.zone,
                node_count=config.cluster.node_count,
                worker_ordinal=assignment.host_ordinal,
                full_image_id=config.image_digest,
                command=_phase_command(
                    paths,
                    "sampling-host",
                    phase_id=manifest.phase_id,
                    host_ordinal=assignment.host_ordinal,
                ),
                env=env,
                foreground=False,
                name=SAMPLER_CONTAINER_NAME,
            )

    def wait_for_sampling_phase(
        self,
        config: AlternatingRLConfig,
        state: AlternatingRunState,
        manifest: SamplingManifest,
        paths: AlternatingRunPaths,
    ) -> None:
        del state
        deadline = time.time() + DEFAULT_SAMPLING_TIMEOUT
        sampling_start = time.time()
        missing_container_polls: dict[int, int] = {
            assignment.host_ordinal: 0 for assignment in manifest.host_assignments
        }
        while time.time() < deadline:
            written_statuses = _written_sampling_statuses(paths, manifest)
            failure = _sampling_phase_failure(written_statuses)
            if failure is not None:
                raise RuntimeError(
                    "sampling host reported failure before phase completion: "
                    f"phase_id={failure.phase_id}, host_ordinal={failure.host_ordinal}, "
                    f"error={failure.error_message}"
                )

            written_host_ordinals = {status.host_ordinal for status in written_statuses}
            for assignment in manifest.host_assignments:
                if assignment.host_ordinal in written_host_ordinals:
                    missing_container_polls[assignment.host_ordinal] = 0
                    continue

                container_exists = tpus.container_exists_on_worker(
                    config.cluster.tpu_name,
                    config.cluster.zone,
                    config.cluster.node_count,
                    assignment.host_ordinal,
                    name=SAMPLER_CONTAINER_NAME,
                )
                if container_exists:
                    missing_container_polls[assignment.host_ordinal] = 0
                    continue

                missing_container_polls[assignment.host_ordinal] += 1
                if missing_container_polls[assignment.host_ordinal] >= DEFAULT_MISSING_CONTAINER_GRACE_POLLS:
                    raise RuntimeError(
                        "sampling host container exited before writing status: "
                        f"phase_id={manifest.phase_id}, host_ordinal={assignment.host_ordinal}"
                    )

            if sampling_host_statuses_exist(paths, manifest):
                _log_phase_metrics(
                    paths,
                    manifest.phase_id,
                    sampling_seconds=time.time() - sampling_start,
                )
                return
            time.sleep(DEFAULT_POLL_INTERVAL)
        raise TimeoutError(f"sampling phase {manifest.phase_id} timed out waiting for host completion")

    def update_curriculum_state(
        self,
        config: AlternatingRLConfig,
        state: AlternatingRunState,
        manifest: SamplingManifest,
        paths: AlternatingRunPaths,
    ) -> None:
        update_start = time.time()
        curriculum = create_local_curriculum(config.curriculum, checkpoint_path=paths.curriculum_root)
        for assignment in manifest.host_assignments:
            status: SamplingHostStatusManifest = read_sampling_host_status(
                paths.sampling_host_status_path(manifest.phase_id, assignment.host_ordinal)
            )
            stats = _build_training_rollout_stats(config, status.lesson_rewards)
            curriculum.update_lesson_stats(stats, mode="training", current_step=state.source_global_step)
        curriculum.save_checkpoint(paths.curriculum_root)
        _log_phase_metrics(
            paths,
            manifest.phase_id,
            curriculum_update_seconds=time.time() - update_start,
        )

    def materialize_training_batches(
        self,
        config: AlternatingRLConfig,
        state: AlternatingRunState,
        manifest: SamplingManifest,
        paths: AlternatingRunPaths,
    ) -> str:
        del manifest
        tpus.stop_container_all_workers(
            config.cluster.tpu_name,
            config.cluster.zone,
            config.cluster.node_count,
            name=SAMPLER_CONTAINER_NAME,
        )
        materialization_start = time.time()
        tpus.run_container_on_worker(
            tpu_name=config.cluster.tpu_name,
            zone=config.cluster.zone,
            node_count=config.cluster.node_count,
            worker_ordinal=0,
            full_image_id=config.image_digest,
            command=_phase_command(paths, "materialize", phase_id=state.phase_id),
            env=_common_phase_env(
                config,
                compilation_cache_dir=config.caches.training_compilation_cache_dir,
            ),
            foreground=True,
            name=MATERIALIZER_CONTAINER_NAME,
        )
        tpus.stop_container_on_worker(
            config.cluster.tpu_name,
            config.cluster.zone,
            config.cluster.node_count,
            0,
            name=MATERIALIZER_CONTAINER_NAME,
        )
        _log_phase_metrics(
            paths,
            state.phase_id,
            materialization_seconds=time.time() - materialization_start,
        )
        return paths.materialized_manifest_path(state.phase_id)

    def run_training_phase(
        self,
        config: AlternatingRLConfig,
        state: AlternatingRunState,
        manifest,
        paths: AlternatingRunPaths,
    ) -> str:
        del manifest
        tpus.stop_container_all_workers(
            config.cluster.tpu_name,
            config.cluster.zone,
            config.cluster.node_count,
            name=TRAINER_CONTAINER_NAME,
        )
        tpus.probe_tpu_all_workers_health(
            config.cluster.tpu_name,
            config.cluster.zone,
            config.cluster.node_count,
        )
        tpus.run_container_all_workers(
            tpu_name=config.cluster.tpu_name,
            zone=config.cluster.zone,
            node_count=config.cluster.node_count,
            full_image_id=config.image_digest,
            command=_phase_command(paths, "train-phase", phase_id=state.phase_id),
            env=_common_phase_env(
                config,
                compilation_cache_dir=config.caches.training_compilation_cache_dir,
            ),
            foreground=True,
            name=TRAINER_CONTAINER_NAME,
        )
        tpus.stop_container_all_workers(
            config.cluster.tpu_name,
            config.cluster.zone,
            config.cluster.node_count,
            name=TRAINER_CONTAINER_NAME,
        )
        return paths.policy_manifest_path(state.policy_version + 1)


def run_prepare_sampling_from_config_path(config_path: str, phase_id: int) -> None:
    """Local helper for the remote `prepare-sampling` subcommand."""
    config = read_pickle(config_path)
    paths = AlternatingRunPaths.from_config(config)
    prepare_sampling_phase(config, paths, phase_id)


def run_sampling_host_from_config_path(config_path: str, phase_id: int, host_ordinal: int) -> None:
    """Local helper for the remote `sampling-host` subcommand."""
    config = read_pickle(config_path)
    paths = AlternatingRunPaths.from_config(config)
    run_sampling_host(config, paths, phase_id, host_ordinal)


def run_materialization_from_config_path(config_path: str, phase_id: int) -> str:
    """Local helper for the remote `materialize` subcommand."""
    config = read_pickle(config_path)
    paths = AlternatingRunPaths.from_config(config)
    return run_materialization(config, paths, phase_id)


def run_training_phase_from_config_path(config_path: str, phase_id: int) -> str:
    """Local helper for the remote `train-phase` subcommand."""
    from marin.rl.alternating.state import read_materialized_batches_manifest, read_run_state

    config = read_pickle(config_path)
    paths = AlternatingRunPaths.from_config(config)
    state = read_run_state(paths.run_state_path)
    if state.phase_id != phase_id:
        logger.warning("training phase requested for phase %d but run_state is at phase %d", phase_id, state.phase_id)
    if not exists(paths.materialized_manifest_path(phase_id)):
        raise FileNotFoundError(f"missing materialized manifest for phase {phase_id}")
    manifest = read_materialized_batches_manifest(paths.materialized_manifest_path(phase_id))
    return run_training_phase_local(config, paths, state, manifest)


def run_export_policy_from_config_path(config_path: str, phase_id: int) -> str:
    """Local helper for the remote `export-policy` subcommand."""
    from marin.rl.alternating.state import read_materialized_batches_manifest, read_run_state

    config = read_pickle(config_path)
    paths = AlternatingRunPaths.from_config(config)
    state = read_run_state(paths.run_state_path)
    if state.phase_id != phase_id:
        logger.warning("export-only requested for phase %d but run_state is at phase %d", phase_id, state.phase_id)
    if not exists(paths.materialized_manifest_path(phase_id)):
        raise FileNotFoundError(f"missing materialized manifest for phase {phase_id}")
    manifest = read_materialized_batches_manifest(paths.materialized_manifest_path(phase_id))
    return export_policy_only(config, paths, state, manifest)
