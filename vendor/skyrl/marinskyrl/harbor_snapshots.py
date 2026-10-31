"""Bind an approved image and its explicit resource overrides to a Harbor trial."""

from collections.abc import Mapping

from harbor_config.models.trial.config import EnvironmentConfig


def approved_snapshot_environment(
    environment: EnvironmentConfig,
    source_key: str,
    snapshots: Mapping[str, str],
    overrides: Mapping[str, EnvironmentConfig],
) -> EnvironmentConfig:
    if source_key not in snapshots:
        raise ValueError(f"Harbor environment has no approved snapshot: {source_key}")
    effective = environment.model_copy(
        update=overrides[source_key].model_dump(exclude_unset=True) if source_key in overrides else {}
    )
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
