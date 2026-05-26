# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint executed by one regional Zephyr job.

The meta-coordinator submits one ``regional_job.main(spec)`` Fray job per
region. Inside, this module:

1. Validates that the model and input paths live in the worker's region
   (the canonical results path is intentionally cross-region and is
   excluded from the check).
2. Constructs a `ZephyrContext` with per-context heartbeat / failure caps
   from the `InferenceConfig`, and a worker ``EnvironmentConfig`` carrying
   the uv extras and JAX / vLLM XLA-cache env vars for the TPU worker
   process. The compile cache only takes effect when its env vars are set
   on the worker, not on this CPU coordinator.
3. Builds the inference pipeline and calls `ctx.execute`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from fray import ResourceConfig, current_client
from fray.types import create_environment
from rigging.filesystem import check_gcs_paths_same_region
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner

from .compile_cache import configure_env, resolve_cache_uri
from .config import ModelSpec, SamplingParams
from .pipeline import assign_shard_ids, build_dataset, rotate_for_region

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegionalJobSpec:
    """Serializable config for one regional inference job.

    Constructed by the meta-coordinator from `InferenceConfig` plus the
    region-specific bits (``region``, the sorted input file list, the
    canonical ``results_uri``). Passed via cloudpickle to the regional Fray
    job's entrypoint.
    """

    region: str
    results_uri: str
    input_files: tuple[str, ...]
    model_spec: ModelSpec
    sampling: SamplingParams
    job_name: str
    run_id: str
    tpu_shapes: tuple[str, ...]
    max_workers: int
    worker_preemptible: bool
    heartbeat_timeout: float
    max_shard_failures: int
    max_shard_infra_failures: int
    chunk_size: int
    compile_cache_uri_template: str | None
    worker_extras: tuple[str, ...]


def main(spec: RegionalJobSpec) -> None:
    """Run the regional inference job to completion (blocks)."""
    logger.info(
        "Starting regional inference job: region=%s, %d input files, results_uri=%s",
        spec.region,
        len(spec.input_files),
        spec.results_uri,
    )

    _validate_region(spec)

    work_items = assign_shard_ids(spec.input_files)
    rotated = rotate_for_region(work_items, spec.region)

    ctx = _build_context(spec)
    dataset = build_dataset(
        rotated,
        model_spec=spec.model_spec,
        sampling=spec.sampling,
        region=spec.region,
        results_uri=spec.results_uri,
    )
    ctx.execute(dataset)
    logger.info("Regional inference job complete: region=%s, %d shards", spec.region, len(rotated))


def _validate_region(spec: RegionalJobSpec) -> None:
    """Crash early if model or input paths are not in the worker's region.

    The canonical ``results_uri`` is intentionally cross-region when
    ``results_region != region``; we deliberately exclude it from the check.
    """
    resolved_model = spec.model_spec.resolve_for_region(spec.region)
    check_gcs_paths_same_region(
        {"model": resolved_model, "input_files": list(spec.input_files)},
        local_ok=False,
        region=spec.region,
    )


def _build_worker_env_vars(spec: RegionalJobSpec) -> dict[str, str]:
    """Compute env vars to set on the TPU worker process.

    Currently just the JAX / vLLM XLA compile-cache vars. Empty dict when the
    cache is disabled (template="") so the caller can skip building a
    ``worker_environment`` if there's also nothing else to inject.
    """
    env_vars: dict[str, str] = {}
    cache_uri = resolve_cache_uri(spec.model_spec, spec.region, spec.compile_cache_uri_template)
    configure_env(cache_uri, env=env_vars)
    return env_vars


def _build_context(spec: RegionalJobSpec) -> ZephyrContext:
    client = current_client()
    worker_resources = ResourceConfig.with_tpu(
        list(spec.tpu_shapes), preemptible=spec.worker_preemptible, regions=[spec.region]
    )
    coordinator_resources = ResourceConfig(cpu=0.5, ram="2g", preemptible=False, regions=[spec.region])
    worker_env_vars = _build_worker_env_vars(spec)
    if spec.worker_extras or worker_env_vars:
        worker_environment = create_environment(extras=list(spec.worker_extras), env_vars=worker_env_vars)
    else:
        worker_environment = None
    return ZephyrContext(
        client=client,
        max_workers=spec.max_workers,
        resources=worker_resources,
        coordinator_resources=coordinator_resources,
        stage_runner_factory=InlineRunner,
        heartbeat_timeout=spec.heartbeat_timeout,
        max_shard_failures=spec.max_shard_failures,
        max_shard_infra_failures=spec.max_shard_infra_failures,
        worker_environment=worker_environment,
        name=f"{spec.job_name}-{spec.region}-{spec.run_id}",
    )
