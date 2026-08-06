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


def validate_merge_storage_region(
    paths: MergeStoragePaths,
    *,
    local_ok: bool,
    region: str | None = None,
    region_getter: Callable[[], str | None] | None = None,
    path_checker: Callable[[str, str, str, bool], None] | None = None,
) -> None:
    """Fail before work begins unless every material merge path is local or same-region GCS."""
    if not local_ok:
        for field in dataclasses.fields(paths):
            value = getattr(paths, field.name)
            if not value.startswith("gs://"):
                raise ValueError(f"{field.name} must be a GCS path on accelerator workers, got {value}")
    check_gcs_paths_same_region(
        paths,
        local_ok=local_ok,
        region=region,
        region_getter=region_getter,
        path_checker=path_checker,
        skip_if_prefix_contains=(),
    )


__all__ = ["MergeStoragePaths", "validate_merge_storage_region"]
