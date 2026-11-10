# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serving cache identity and the CPU jobs that fill it before accelerators are reserved."""

import logging
import uuid
from dataclasses import dataclass, replace

from fray.client import Client, JobHandle, wait_all
from fray.types import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig
from levanter.model_cache import HfModelCacheEntry, cache_hf_model, hf_model_cache_entry
from rigging.timing import Duration

from marin.inference.config import InferenceEngineConfig, IrisConfig, ServedModelConfig, VllmEngineConfig

logger = logging.getLogger(__name__)

MODEL_CACHE_PREFIX = "quick-serve-models"
_PREPARE_JOB_TIMEOUT_HOURS = 4
_PREPARE_WAIT_TIMEOUT = 8 * 3600
_PREPARE_CPU = 4
_PREPARE_RAM = "8g"
_PREPARE_DISK = "40g"


@dataclass(frozen=True)
class _PinnedModels:
    model: ServedModelConfig
    engine: InferenceEngineConfig
    entries: tuple[HfModelCacheEntry, ...]


class ModelPreparationError(RuntimeError):
    """Preparing a serving model failed before any accelerator job was submitted."""

    def __init__(self, message: str, jobs: tuple[JobHandle, ...]):
        super().__init__(message)
        self.jobs = jobs


def model_cache_entry(model: str, cache_ttl_days: int, revision: str | None = None) -> HfModelCacheEntry | None:
    """Return the serving cache entry without downloading model weights."""
    if revision is not None:
        model = f"{model}@{revision}"
    return hf_model_cache_entry(model, cache_ttl_days=cache_ttl_days, cache_prefix=MODEL_CACHE_PREFIX)


def populate_model_cache(repo: str, revision: str, cache_ttl_days: int) -> str:
    """Mirror ``repo@revision`` into the serving cache of the storage this process runs against.

    The cache path depends on the process's storage prefix. Returns immediately when the entry
    is already complete.
    """
    entry = model_cache_entry(repo, cache_ttl_days, revision)
    if entry is None:
        raise ValueError(f"{repo}@{revision} is not a cacheable Hugging Face model (cache_ttl_days={cache_ttl_days})")
    return cache_hf_model(entry.path, entry.repo, revision=entry.revision)


def prepare_hf_models(
    model: ServedModelConfig,
    engine: InferenceEngineConfig,
    iris: IrisConfig,
    *,
    client: Client,
    environment: EnvironmentConfig,
) -> tuple[ServedModelConfig, InferenceEngineConfig]:
    """Pin Hugging Face models to commits and fill their serving cache from CPU jobs.

    Covers the served model and a vLLM speculative draft model. Each model gets one CPU job using
    the region and target cluster constraints in ``iris.worker_resources``. Call it before submitting
    accelerator jobs and serve the returned configs. Other models, and any model when caching is
    disabled, pass through unchanged.

    Raises:
        ModelPreparationError: resolving a revision, submitting, or a prepare job failed. Submitted
            jobs are terminated first.
    """
    try:
        pinned = _pinned_models(model, engine, iris.cache_ttl_days)
    except Exception as exc:
        raise ModelPreparationError(f"Model cache preparation failed: {exc}", jobs=()) from exc
    jobs: list[JobHandle] = []
    try:
        for entry in pinned.entries:
            job = client.submit(_prepare_job_request(entry, iris, environment))
            logger.info("preparing %s@%s in job %s", entry.repo, entry.revision, job.job_id)
            jobs.append(job)
        wait_all(jobs, timeout=_PREPARE_WAIT_TIMEOUT)
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


def _pinned_models(model: ServedModelConfig, engine: InferenceEngineConfig, cache_ttl_days: int) -> _PinnedModels:
    entries: list[HfModelCacheEntry] = []
    model_entry = model_cache_entry(model.weights, cache_ttl_days, model.revision)
    if model_entry is not None:
        entries.append(model_entry)
        model = replace(model, weights=model_entry.repo, revision=model_entry.revision, api_model=model.model_id)
    if isinstance(engine, VllmEngineConfig) and engine.speculative is not None:
        speculative = engine.speculative
        draft_entry = model_cache_entry(speculative.model.uri, cache_ttl_days)
        if draft_entry is not None:
            entries.append(draft_entry)
            draft_model = replace(speculative.model, uri=f"{draft_entry.repo}@{draft_entry.revision}")
            engine = replace(engine, speculative=replace(speculative, model=draft_model))
    return _PinnedModels(model, engine, tuple(entries))


def _prepare_job_request(entry: HfModelCacheEntry, iris: IrisConfig, environment: EnvironmentConfig) -> JobRequest:
    worker_resources = iris.worker_resources
    return JobRequest(
        name=f"prepare-model-{uuid.uuid4().hex}",
        entrypoint=Entrypoint.from_callable(
            populate_model_cache, args=(entry.repo, entry.revision, iris.cache_ttl_days)
        ),
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
        timeout=Duration.from_hours(_PREPARE_JOB_TIMEOUT_HOURS),
    )
