# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Meta-coordinator: orchestrates per-region inference jobs.

`run_meta_coordinator` submits one Fray job per region — each running
`regional_job.main` — and waits for all of them to finish. Failure of any
single region is **non-fatal**: the run is considered successful as long as
every shard has an output file in the results bucket at the end. Surviving
regions cover for failed ones via the per-region rotation + skip_existing
semantics in the pipeline.

The meta-coordinator itself runs in the caller's process. When `inference()`
is called from an ExecutorStep (the expected pattern), the ExecutorStep's
own Fray job hosts the meta-coordinator; there is no extra wrapping job.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

from fray import ResourceConfig, current_client
from fray.client import JobHandle
from fray.types import Entrypoint, JobRequest

from .config import InferenceConfig, ModelSpec
from .output import InferenceResult, compute_missing_shards
from .regional_job import RegionalJobSpec
from .regional_job import main as regional_main

logger = logging.getLogger(__name__)


def run_meta_coordinator(
    *,
    config: InferenceConfig,
    input_files: Sequence[str],
    model_spec: ModelSpec,
    results_uri: str,
    run_id: str,
) -> InferenceResult:
    """Submit per-region jobs, wait, return an `InferenceResult`."""
    client = current_client()
    handles: list[tuple[str, JobHandle]] = []
    for region in config.regions:
        spec = _make_regional_spec(
            config=config,
            region=region,
            input_files=input_files,
            model_spec=model_spec,
            results_uri=results_uri,
            run_id=run_id,
        )
        handle = client.submit(_make_job_request(spec, config))
        logger.info(
            "Submitted regional inference job: region=%s, job_id=%s, name=%s",
            region,
            handle.job_id,
            spec.job_name,
        )
        handles.append((region, handle))

    for region, handle in handles:
        try:
            status = handle.wait(raise_on_failure=False)
            logger.info("Regional job done: region=%s, status=%s", region, status)
        except Exception:
            logger.warning("Regional job for %s raised on wait; continuing.", region, exc_info=True)

    missing = compute_missing_shards(results_uri, total_shards=len(input_files))
    if missing:
        logger.warning(
            "Run completed with %d missing shards out of %d total.",
            len(missing),
            len(input_files),
        )
    return InferenceResult(
        results_uri=results_uri,
        results_region=config.results_region,
        missing_shards=missing,
    )


def _make_regional_spec(
    *,
    config: InferenceConfig,
    region: str,
    input_files: Sequence[str],
    model_spec: ModelSpec,
    results_uri: str,
    run_id: str,
) -> RegionalJobSpec:
    return RegionalJobSpec(
        region=region,
        results_uri=results_uri,
        input_files=tuple(input_files),
        model_spec=model_spec,
        sampling=config.sampling,
        job_name=config.job_name,
        run_id=run_id,
        tpu_shapes=tuple(config.tpu_shapes),
        max_workers=config.max_workers_per_region,
        worker_preemptible=config.worker_preemptible,
        heartbeat_timeout=config.heartbeat_timeout,
        max_shard_failures=config.max_shard_failures,
        max_shard_infra_failures=config.max_shard_infra_failures,
        chunk_size=config.chunk_size,
        compile_cache_uri_template=config.compile_cache_uri_template,
        worker_extras=tuple(config.worker_extras),
    )


def _make_job_request(spec: RegionalJobSpec, config: InferenceConfig) -> JobRequest:
    name = f"{spec.job_name}-{spec.region}-{spec.run_id}"
    # The regional job is a small CPU coordinator that itself spawns Zephyr's
    # coordinator + worker jobs. Run it non-preemptible so we don't lose the
    # job-monitoring layer mid-run.
    resources = ResourceConfig(cpu=0.5, ram="2g", preemptible=False, regions=[spec.region])
    return JobRequest(
        name=name,
        entrypoint=Entrypoint.from_callable(regional_main, args=(spec,)),
        resources=resources,
        # Zephyr's per-shard retries handle preemption inside; if the regional
        # coordinator itself dies we let it retry once to handle CPU-node hiccups
        # but otherwise treat regional failure as non-fatal at the run level.
        max_retries_preemption=1,
    )
