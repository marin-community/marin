# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-host sampling process for alternating RL.

Each TPU host runs one instance of this process during the sampling phase.
It loads one frozen policy, generates rollouts until its quota is met, and
writes raw rollout files and a host status file.
"""

import logging
import os
import pickle
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import equinox as eqx
import numpy as np
from iris.marin_fs import url_to_fs

from marin.rl.alternating.state import (
    HostStatus,
    PolicyManifest,
    SamplingHostAssignment,
    SamplingManifest,
    read_json_from_path,
    write_json_to_path,
)
from marin.rl.curriculum import CurriculumConfig, Curriculum
from marin.rl.environments import MarinEnv
from marin.rl.environments.base import load_environment_from_spec
from marin.rl.environments.inference_ctx import vLLMInferenceContextConfig, vLLMInferenceContext
from marin.rl.types import RolloutBatch, RolloutGroup, RolloutMetadata

logger = logging.getLogger(__name__)


@dataclass
class SamplingHostConfig:
    """Configuration for one sampling host process."""

    manifest_path: str
    host_ordinal: int
    curriculum_config: CurriculumConfig
    model_name: str
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float = 0.90
    system_prompt: str | None = None
    seed: int = 42


def run_sampling_host(config: SamplingHostConfig) -> None:
    """Run the sampling process on one TPU host.

    1. Read manifests and load frozen policy
    2. Run eval quota
    3. Run train quota
    4. Write host status
    """
    started_at = datetime.now(timezone.utc).isoformat()
    logger.info("Sampling host %d starting, manifest=%s", config.host_ordinal, config.manifest_path)

    manifest = SamplingManifest.from_json(read_json_from_path(config.manifest_path))
    policy_manifest = PolicyManifest.from_json(read_json_from_path(manifest.policy_manifest_path))
    assignment = _find_assignment(manifest, config.host_ordinal)

    # Build vLLM inference context with the frozen policy
    from vllm import SamplingParams as VllmSamplingParams

    inference_config = vLLMInferenceContextConfig(
        model_name=config.model_name,
        max_model_len=config.max_model_len,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        sampling_params=VllmSamplingParams(),
        enable_fast_bootstrap=True,
        bootstrap_checkpoint_path=policy_manifest.hf_export_path,
    )
    policy_ctx = vLLMInferenceContext(inference_config)
    logger.info("Policy loaded for version %d from %s", manifest.policy_version, policy_manifest.hf_export_path)

    # Build local curriculum for lesson sampling (frozen weights)
    curriculum = Curriculum(config.curriculum_config)
    if manifest.curriculum_state_path:
        try:
            checkpoint_dir = os.path.dirname(manifest.curriculum_state_path)
            filename = os.path.basename(manifest.curriculum_state_path)
            curriculum.restore_checkpoint(checkpoint_dir, filename)
        except Exception as e:
            logger.warning("Could not restore curriculum state: %s", e)

    # Load environments
    environments: dict[str, MarinEnv] = {}

    host_output_root = manifest.output_root + f"/host_{config.host_ordinal:03d}"
    rng = np.random.default_rng(assignment.seed)

    eval_groups_written = 0
    train_groups_written = 0
    error = None

    try:
        # Phase 1: Eval prelude
        if assignment.target_eval_groups > 0:
            eval_groups_written = _run_quota(
                policy_ctx=policy_ctx,
                curriculum_config=config.curriculum_config,
                curriculum=curriculum,
                frozen_weights=manifest.frozen_lesson_weights,
                environments=environments,
                target_groups=assignment.target_eval_groups,
                output_dir=f"{host_output_root}/eval",
                mode="eval",
                rng=rng,
                manifest=manifest,
                assignment=assignment,
                system_prompt=config.system_prompt,
            )
            logger.info("Eval prelude done: %d groups written", eval_groups_written)

        # Phase 2: Train rollout generation
        if assignment.target_train_groups > 0:
            train_groups_written = _run_quota(
                policy_ctx=policy_ctx,
                curriculum_config=config.curriculum_config,
                curriculum=curriculum,
                frozen_weights=manifest.frozen_lesson_weights,
                environments=environments,
                target_groups=assignment.target_train_groups,
                output_dir=f"{host_output_root}/train",
                mode="training",
                rng=rng,
                manifest=manifest,
                assignment=assignment,
                system_prompt=config.system_prompt,
            )
            logger.info("Train quota done: %d groups written", train_groups_written)

        success = True
    except Exception as e:
        logger.exception("Sampling host %d failed", config.host_ordinal)
        success = False
        error = str(e)

    # Write host status
    finished_at = datetime.now(timezone.utc).isoformat()
    status = HostStatus(
        host_ordinal=config.host_ordinal,
        phase_id=manifest.phase_id,
        policy_version=manifest.policy_version,
        eval_groups_written=eval_groups_written,
        train_groups_written=train_groups_written,
        success=success,
        started_at=started_at,
        finished_at=finished_at,
        error=error,
    )
    status_path = f"{host_output_root}/status.json"
    write_json_to_path(status_path, status.to_json())
    logger.info("Host status written to %s (success=%s)", status_path, success)

    # Shutdown inference context
    policy_ctx.shutdown()

    if not success:
        raise RuntimeError(f"Sampling host {config.host_ordinal} failed: {error}")


def _find_assignment(manifest: SamplingManifest, host_ordinal: int) -> SamplingHostAssignment:
    for ha in manifest.host_assignments:
        if ha.host_ordinal == host_ordinal:
            return ha
    raise ValueError(f"No assignment found for host_ordinal={host_ordinal} in manifest")


def _sample_lesson(frozen_weights: dict[str, float], rng: np.random.Generator) -> str:
    """Sample a lesson from the frozen weight distribution."""
    lesson_ids = list(frozen_weights.keys())
    probs = np.array([frozen_weights[lid] for lid in lesson_ids])
    probs = probs / probs.sum()
    idx = rng.choice(len(lesson_ids), p=probs)
    return lesson_ids[idx]


def _run_quota(
    policy_ctx: vLLMInferenceContext,
    curriculum_config: CurriculumConfig,
    curriculum: Curriculum,
    frozen_weights: dict[str, float],
    environments: dict[str, MarinEnv],
    target_groups: int,
    output_dir: str,
    mode: str,
    rng: np.random.Generator,
    manifest: SamplingManifest,
    assignment: SamplingHostAssignment,
    system_prompt: str | None,
) -> int:
    """Generate rollouts until the target group count is met."""
    fs, _ = url_to_fs(output_dir)
    fs.makedirs(output_dir, exist_ok=True)

    groups_written = 0
    file_counter = 0

    while groups_written < target_groups:
        lesson_id = _sample_lesson(frozen_weights, rng)
        lesson_config = curriculum_config.lessons[lesson_id]

        if lesson_id not in environments:
            environments[lesson_id] = load_environment_from_spec(lesson_config.env_config)
        env = environments[lesson_id]

        n_examples = lesson_config.sampling_params.n_prompts
        n_generations = lesson_config.sampling_params.n_generations_per_prompt
        temperature = lesson_config.sampling_params.temperature
        top_k = lesson_config.sampling_params.top_k
        stop_tokens = lesson_config.sampling_params.stop_tokens
        max_tokens = lesson_config.sampling_params.max_output_tokens

        rollout_groups, _metrics = env.sample(
            inference_ctx=policy_ctx,
            n_examples=n_examples,
            n_generations=n_generations,
            temperature=temperature,
            prng_key=rng,
            mode=mode,
            max_tokens=max_tokens,
            top_k=top_k,
            stop=stop_tokens,
            system_prompt=system_prompt,
        )

        if not rollout_groups:
            logger.warning("No rollouts generated for lesson %s", lesson_id)
            continue

        batch_metadata = RolloutMetadata(
            worker_id=f"{socket.gethostname()}_{os.getpid()}",
            timestamp=time.time(),
            weight_step=manifest.policy_version,
            policy_version=manifest.policy_version,
            phase_id=manifest.phase_id,
            source_global_step=-1,
        )

        groups_with_meta = []
        for group in rollout_groups:
            rollouts_with_meta = []
            for rollout in group.rollouts:
                rollout_with_meta = eqx.tree_at(lambda r: r.metadata, rollout, batch_metadata)
                rollouts_with_meta.append(rollout_with_meta)
            groups_with_meta.append(RolloutGroup(rollouts=rollouts_with_meta))

        rollout_batch = RolloutBatch(groups=groups_with_meta, metadata=batch_metadata)

        # Write batch to file
        out_path = f"{output_dir}/{file_counter:06d}.pkl"
        with fs.open(out_path, "wb") as f:
            pickle.dump(rollout_batch, f)

        batch_groups = len(rollout_batch.groups)
        groups_written += batch_groups
        file_counter += 1
        logger.info(
            "Wrote %d groups (%d/%d total) to %s",
            batch_groups,
            groups_written,
            target_groups,
            out_path,
        )

    return groups_written
