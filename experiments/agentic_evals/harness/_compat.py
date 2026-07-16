"""Compatibility shim for Harbor dataset-config + orchestrator classes.

Copied from OT-Agent ``scripts/harbor/_harbor_compat.py``.

Hides the difference between legacy Harbor (distinct ``LocalDatasetConfig`` /
``RegistryDatasetConfig`` + nested ``OrchestratorConfig``) and unified Harbor
(one ``DatasetConfig`` + flat top-level fields) so callers don't need to know
which version is installed.
"""

from __future__ import annotations

import asyncio
import inspect

try:
    from harbor.models.job.config import (  # type: ignore[attr-defined]
        LocalDatasetConfig,
        RegistryDatasetConfig,
    )
    _UNIFIED_DATASET_CONFIG = False
except ImportError:
    from harbor.models.job.config import DatasetConfig as _DatasetConfig
    LocalDatasetConfig = _DatasetConfig
    RegistryDatasetConfig = _DatasetConfig
    _UNIFIED_DATASET_CONFIG = True


def is_local_dataset(dataset) -> bool:
    if hasattr(dataset, "is_local") and callable(dataset.is_local):
        return bool(dataset.is_local())
    return isinstance(dataset, LocalDatasetConfig)


# --- Orchestrator concept breaking-change shim ---

try:
    from harbor.orchestrators.base import OrchestratorEvent  # type: ignore[import-not-found]
    TrialEvent = OrchestratorEvent
    TRIAL_COMPLETED_EVENT = OrchestratorEvent.TRIAL_COMPLETED
    _UNIFIED_ORCHESTRATOR = False
except ImportError:
    from harbor.trial.hooks import TrialEvent  # type: ignore[attr-defined]
    OrchestratorEvent = TrialEvent  # type: ignore[assignment]
    TRIAL_COMPLETED_EVENT = TrialEvent.END
    _UNIFIED_ORCHESTRATOR = True


try:
    from harbor.models.job.config import OrchestratorConfig  # type: ignore[attr-defined]
    _HAS_ORCHESTRATOR_CONFIG = True
except ImportError:
    OrchestratorConfig = None  # type: ignore[assignment, misc]
    _HAS_ORCHESTRATOR_CONFIG = False


def build_job_config_kwargs(
    *,
    n_concurrent_trials: int | None = None,
    quiet: bool | None = None,
    retry: object | None = None,
    **other_kwargs,
) -> dict:
    kwargs: dict = dict(other_kwargs)
    if _HAS_ORCHESTRATOR_CONFIG and OrchestratorConfig is not None:
        orch_kwargs: dict = {}
        if n_concurrent_trials is not None:
            orch_kwargs["n_concurrent_trials"] = n_concurrent_trials
        if quiet is not None:
            orch_kwargs["quiet"] = quiet
        if retry is not None:
            orch_kwargs["retry"] = retry
        if orch_kwargs:
            kwargs["orchestrator"] = OrchestratorConfig(**orch_kwargs)
    else:
        if n_concurrent_trials is not None:
            kwargs["n_concurrent_trials"] = n_concurrent_trials
        if quiet is not None:
            kwargs["quiet"] = quiet
        if retry is not None:
            kwargs["retry"] = retry
    return kwargs


def set_orchestrator_field(config, field: str, value) -> None:
    orch = getattr(config, "orchestrator", None)
    if orch is not None:
        setattr(orch, field, value)
        return
    setattr(config, field, value)


def get_orchestrator_field(config, field: str, default=None):
    if isinstance(config, dict):
        orch = config.get("orchestrator")
        if isinstance(orch, dict) and field in orch:
            return orch[field]
        if field in config:
            return config[field]
        return default

    orch = getattr(config, "orchestrator", None)
    if orch is not None:
        v = getattr(orch, field, None)
        if v is not None:
            return v
    return getattr(config, field, default)


def add_trial_completed_hook(job, callback) -> None:
    if hasattr(job, "on_trial_completed") and callable(job.on_trial_completed):
        job.on_trial_completed(callback)
        return
    if hasattr(job, "add_hook") and callable(job.add_hook):
        job.add_hook(TRIAL_COMPLETED_EVENT, callback)
        return
    orch = getattr(job, "_orchestrator", None)
    if orch is not None and hasattr(orch, "add_hook"):
        orch.add_hook(TRIAL_COMPLETED_EVENT, callback)
        return
    raise RuntimeError(
        "Unable to attach trial-completed hook: Job exposes neither "
        "on_trial_completed nor add_hook nor _orchestrator.add_hook."
    )


def _uses_async_factory(JobCls) -> bool:
    create = getattr(JobCls, "create", None)
    if create is None:
        return False
    return inspect.iscoroutinefunction(create)


def create_job(JobCls, config, **kwargs):
    if _uses_async_factory(JobCls):
        return asyncio.run(JobCls.create(config))
    return JobCls(config, **kwargs)


async def create_job_async(JobCls, config, **kwargs):
    if _uses_async_factory(JobCls):
        return await JobCls.create(config)
    return JobCls(config, **kwargs)


__all__ = [
    "LocalDatasetConfig",
    "RegistryDatasetConfig",
    "is_local_dataset",
    "_UNIFIED_DATASET_CONFIG",
    "OrchestratorEvent",
    "TrialEvent",
    "TRIAL_COMPLETED_EVENT",
    "OrchestratorConfig",
    "build_job_config_kwargs",
    "set_orchestrator_field",
    "get_orchestrator_field",
    "add_trial_completed_hook",
    "create_job",
    "create_job_async",
    "_UNIFIED_ORCHESTRATOR",
]
