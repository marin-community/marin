# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.grug.moe.merge_storage import MergeStoragePaths, validate_merge_storage_region


def _paths() -> MergeStoragePaths:
    return MergeStoragePaths(
        teacher_checkpoint="gs://marin-us-central1/xem/teacher",
        calibration="gs://marin-us-central1/xem/calibration",
        converted_checkpoint="gs://marin-us-central1/xem/converted",
        recovery_output="gs://marin-us-central1/xem/recovery",
    )


def test_merge_storage_checks_every_material_path_without_source_skips():
    checked = []

    def checker(key: str, path: str, region: str, local_ok: bool) -> None:
        checked.append((key, path, region, local_ok))

    validate_merge_storage_region(
        _paths(),
        local_ok=False,
        region="us-central1",
        path_checker=checker,
    )

    assert {key for key, _, _, _ in checked} == {
        "teacher_checkpoint",
        "calibration",
        "converted_checkpoint",
        "recovery_output",
    }
    assert all(region == "us-central1" and not local_ok for _, _, region, local_ok in checked)


def test_merge_storage_rejects_local_artifacts_on_accelerator_workers():
    paths = _paths()
    local_calibration = MergeStoragePaths(
        teacher_checkpoint=paths.teacher_checkpoint,
        calibration="/tmp/calibration",
        converted_checkpoint=paths.converted_checkpoint,
        recovery_output=paths.recovery_output,
    )

    with pytest.raises(ValueError):
        validate_merge_storage_region(local_calibration, local_ok=False, region="us-central1")


def test_merge_storage_allows_all_local_smoke_artifacts():
    paths = MergeStoragePaths(
        teacher_checkpoint="/tmp/teacher",
        calibration="/tmp/calibration",
        converted_checkpoint="/tmp/converted",
        recovery_output="/tmp/recovery",
    )

    region_lookups = 0

    def region_getter() -> None:
        nonlocal region_lookups
        region_lookups += 1

    validate_merge_storage_region(paths, local_ok=True, region=None, region_getter=region_getter)

    assert region_lookups == 1
