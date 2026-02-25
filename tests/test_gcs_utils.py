# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from marin.utilities.gcs_utils import check_gcs_paths_same_region


def test_check_gcs_paths_same_region_accepts_matching_region():
    config = {"cache_dir": "gs://bucket/path"}

    check_gcs_paths_same_region(
        config,
        local_ok=False,
        region="us-central1",
        path_checker=lambda _key, _path, _region, _local_ok: None,
    )


def test_check_gcs_paths_same_region_raises_for_mismatch():
    config = {"cache_dir": Path("gs://bucket/path")}

    def checker(_key: str, _path: str, _region: str, _local_ok: bool) -> None:
        raise ValueError("not in the same region")

    with pytest.raises(ValueError, match="not in the same region"):
        check_gcs_paths_same_region(
            config,
            local_ok=False,
            region="us-central1",
            path_checker=checker,
        )


def test_check_gcs_paths_same_region_skips_train_source_urls():
    config = {"train_urls": ["gs://bucket/path"], "validation_urls": ["gs://bucket/path"]}

    def checker(_key: str, _path: str, _region: str, _local_ok: bool) -> None:
        raise AssertionError("source URLs should be skipped")

    check_gcs_paths_same_region(
        config,
        local_ok=False,
        region="us-central1",
        path_checker=checker,
    )


def test_check_gcs_paths_same_region_allows_unknown_region_for_local_runs():
    def fail_region_lookup() -> str | None:
        return None

    check_gcs_paths_same_region(
        {"cache_dir": "gs://bucket/path"},
        local_ok=True,
        region_getter=fail_region_lookup,
    )
