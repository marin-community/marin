# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serving cache identity and the CPU jobs that fill it before accelerators are reserved."""

import logging
import uuid
from dataclasses import dataclass, replace

from fray.client import Client, JobHandle, wait_all
from fray.types import ANY_REGION, Entrypoint, EnvironmentConfig, GpuConfig, JobRequest, ResourceConfig, TpuConfig
from iris.cluster.client.job_info import get_job_info
from iris.cluster.constraints import WellKnownAttribute, region_constraint
from levanter.model_cache import CacheMode, resolve_hf_model_ref
from rigging.timing import Duration

from marin.inference.config import InferenceEngineConfig, IrisConfig, ServedModelConfig, VllmEngineConfig
from marin.inference.model_preparation import resolve_model_path

logger = logging.getLogger(__name__)

_PREPARE_TIMEOUT = Duration.from_hours(4)
_PREPARE_CPU = 4
_PREPARE_RAM = "8g"
_PREPARE_DISK = "40g"


@dataclass(frozen=True)
class _PinnedModels:
    model: ServedModelConfig
    engine: InferenceEngineConfig
    entries: tuple[str, ...]


class ModelPreparationError(RuntimeError):
    """Preparing a serving model failed before any accelerator job was submitted."""

    def __init__(self, message: str, jobs: tuple[JobHandle, ...]):
        super().__init__(message)
        self.jobs = jobs


def prepare_hf_models(
    model: ServedModelConfig,
    engine: InferenceEngineConfig,
    iris: IrisConfig,
    *,
    client: Client,
    environment: EnvironmentConfig,
) -> tuple[ServedModelConfig, InferenceEngineConfig]:
    """Pin Hugging Face models to commits and fill their serving cache from CPU jobs.

    Covers the served model and a vLLM speculative draft model. A colocated coordinator checks
    the completion marker first; misses get CPU jobs at the serving placement. Call it before submitting
    accelerator jobs and serve the returned configs. Other models, and any model when caching is
    disabled, pass through unchanged.

    Raises:
        ModelPreparationError: resolving a revision, submitting, or a prepare job failed. Submitted
            jobs are terminated first.
    """
    try:
        pinned = _pinned_models(model, engine, iris.cache_ttl_days, token=environment.secrets.get("HF_TOKEN"))
    except Exception as exc:
        raise ModelPreparationError(f"Model cache preparation failed: {exc}", jobs=()) from exc
    if (
        pinned.entries
        and isinstance(iris.worker_resources.device, GpuConfig)
        and not iris.worker_resources.target_cluster
        and get_job_info() is None
    ):
        logger.warning(
            "Skipping CPU model preparation for unpinned GPU placement: serving may auto-federate "
            "while the CPU job stays local. Set target_cluster to prepare the serving cache."
        )
        return pinned.model, pinned.engine
    resources = iris.worker_resources
    region_is_fixed = (
        len(resources.regions) == 1 and ANY_REGION not in resources.regions
        if resources.regions
        else get_job_info() is not None
    )
    if (
        pinned.entries
        and isinstance(resources.device, TpuConfig)
        and not resources.target_cluster
        and not region_is_fixed
    ):
        logger.warning(
            "Skipping CPU model preparation for unrestricted regional placement: CPU and accelerator "
            "jobs may use different caches. Pin one region to prepare the serving cache."
        )
        return pinned.model, pinned.engine
    jobs: list[JobHandle] = []
    try:
        check_local = bool(pinned.entries) and _coordinator_matches_serving_placement(iris.worker_resources)
        for entry in pinned.entries:
            if check_local and resolve_model_path(entry, iris.cache_ttl_days, cache_mode=CacheMode.READ_ONLY) != entry:
                continue
            job = client.submit(_prepare_job_request(entry, iris, environment))
            logger.info("preparing %s in job %s", entry, job.job_id)
            jobs.append(job)
        wait_all(jobs, timeout=(iris.task_timeout or _PREPARE_TIMEOUT).to_seconds())
    except BaseException as exc:
        # Includes KeyboardInterrupt, so an interactive caller does not leave mirror jobs running.
        for job in jobs:
            try:
                job.terminate()
            except Exception:
                logger.warning("could not terminate model preparation job %s", job.job_id, exc_info=True)
        if not isinstance(exc, Exception):
            raise
        raise ModelPreparationError(f"Model cache preparation failed: {exc}", jobs=tuple(jobs)) from exc
    return pinned.model, pinned.engine


def _pinned_models(
    model: ServedModelConfig, engine: InferenceEngineConfig, cache_ttl_days: int, *, token: str | None
) -> _PinnedModels:
    if cache_ttl_days <= 0:
        return _PinnedModels(model, engine, ())
    entries: list[str] = []
    pinned = resolve_hf_model_ref(f"{model.weights}@{model.revision}" if model.revision else model.weights, token=token)
    if pinned is not None:
        entries.append(pinned)
        repo, _, revision = pinned.partition("@")
        model = replace(model, weights=repo, revision=revision, api_model=model.model_id)
    if isinstance(engine, VllmEngineConfig) and engine.speculative is not None:
        speculative = engine.speculative
        pinned_draft = resolve_hf_model_ref(speculative.model.uri, token=token)
        if pinned_draft is not None:
            entries.append(pinned_draft)
            draft_model = replace(speculative.model, uri=pinned_draft)
            engine = replace(engine, speculative=replace(speculative, model=draft_model))
    return _PinnedModels(model, engine, tuple(dict.fromkeys(entries)))


def _prepare_job_request(entry: str, iris: IrisConfig, environment: EnvironmentConfig) -> JobRequest:
    worker_resources = iris.worker_resources
    return JobRequest(
        name=f"prepare-model-{uuid.uuid4().hex}",
        entrypoint=Entrypoint.from_callable(resolve_model_path, args=(entry, iris.cache_ttl_days)),
        resources=ResourceConfig.with_cpu(
            cpu=_PREPARE_CPU,
            ram=_PREPARE_RAM,
            disk=_PREPARE_DISK,
            preemptible=False,
            regions=worker_resources.regions,
            target_cluster=worker_resources.target_cluster,
        ),
        environment=environment,
        priority=iris.priority,
        timeout=iris.task_timeout or _PREPARE_TIMEOUT,
        max_retries_failure=1,
    )


def _coordinator_matches_serving_placement(resources: ResourceConfig) -> bool:
    if resources.target_cluster is not None:
        return False
    info = get_job_info()
    if info is None or info.worker_region is None:
        return False
    if resources.regions:
        if list(resources.regions) != [info.worker_region]:
            return False
    else:
        # Unset children inherit parent constraints, or the parent's physical region.
        expected = region_constraint([info.worker_region])
        if any(c.key == WellKnownAttribute.REGION and c != expected for c in info.constraints):
            return False
    return True
