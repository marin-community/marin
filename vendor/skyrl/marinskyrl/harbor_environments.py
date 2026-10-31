"""Resolve native Harbor environments and optional explicit snapshot approvals."""

from collections.abc import Mapping

from harbor_config.models.trial.config import EnvironmentConfig
from pydantic import BaseModel, ConfigDict


class EnvironmentResourceOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    override_cpus: int | None = None
    override_memory_mb: int | None = None
    override_storage_mb: int | None = None
    override_gpus: int | None = None


def original_trial_environment(
    environment: EnvironmentConfig,
    source_key: str,
    snapshots: Mapping[str, str],
    overrides: Mapping[str, EnvironmentResourceOverrides],
) -> EnvironmentConfig:
    effective = environment.model_copy(
        update=overrides[source_key].model_dump(exclude_unset=True) if source_key in overrides else {}
    )
    if not snapshots:
        return effective
    if source_key not in snapshots:
        raise ValueError(f"Harbor environment has no approved snapshot: {source_key}")
    return effective.model_copy(
        update={
            "force_build": False,
            "kwargs": {
                **effective.kwargs,
                "snapshot_template_name": snapshots[source_key],
                "auto_snapshot": False,
                "assume_global_snapshot": True,
            },
        }
    )
