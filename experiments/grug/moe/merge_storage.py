# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage-region contract for expert calibration, conversion, and recovery."""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

from rigging.filesystem import check_gcs_paths_same_region


@dataclass(frozen=True)
class MergeStoragePaths:
    teacher_checkpoint: str
    calibration: str
    converted_checkpoint: str
    recovery_output: str
    matching: str | None = None
    prefit_checkpoint: str | None = None
    stage_a_output: str | None = None
    stage_b_output: str | None = None


def _material_paths(paths: MergeStoragePaths) -> list[tuple[str, str]]:
    return [
        (field.name, value) for field in dataclasses.fields(paths) if (value := getattr(paths, field.name)) is not None
    ]


def validate_merge_storage_region(
    paths: MergeStoragePaths,
    *,
    local_ok: bool,
    region: str | None = None,
    region_getter: Callable[[], str | None] | None = None,
    path_checker: Callable[[str, str, str, bool], None] | None = None,
) -> None:
    """Fail before work begins unless every material merge path is local or same-region GCS."""
    material_paths = _material_paths(paths)
    if not local_ok:
        for field_name, value in material_paths:
            if not value.startswith("gs://"):
                raise ValueError(f"{field_name} must be a GCS path on accelerator workers, got {value}")
    else:
        gcs_fields = [field_name for field_name, value in material_paths if value.startswith("gs://")]
        local_fields = [field_name for field_name, value in material_paths if not value.startswith("gs://")]
        if gcs_fields and local_fields:
            raise ValueError(
                "local merge smoke paths must be all local or all GCS; "
                f"GCS fields={gcs_fields}, local fields={local_fields}"
            )
    check_gcs_paths_same_region(
        paths,
        local_ok=local_ok,
        region=region,
        region_getter=region_getter,
        path_checker=path_checker,
        skip_if_prefix_contains=(),
    )


__all__ = ["MergeStoragePaths", "validate_merge_storage_region"]
