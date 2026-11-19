# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from experiments.post_training.task_curation.datasets.endless.images import (
    DATASET,
    EXPECTED_IMAGES,
    EXPECTED_TASKS,
    IMAGE_REVISION,
    image_files,
)


def native_image_index():
    bundles = {}
    files = {}
    bundle_ids = []
    for index in range(EXPECTED_IMAGES):
        bundle = f"{index:064x}"
        filename = f"image-{index}.sif"
        prefix = f"cached_images/{bundle}/{filename}"
        digest = f"{index + 1:064x}"
        bundles[bundle] = {"prefix": prefix, "files": {filename: {"bytes": 123, "sha256": digest}}}
        files[prefix + "/" + filename] = {
            "bundle_id": bundle,
            "filename": filename,
            "bytes": 123,
            "sha256": digest,
        }
        bundle_ids.append(bundle)
    offset = 0
    tasks = {}
    for folder, count in EXPECTED_TASKS.items():
        tasks[folder] = {
            f"task-{row}": {"environments": [{"bundle": bundle_ids[(offset + row) % EXPECTED_IMAGES]}]}
            for row in range(count)
        }
        offset += count
    return {"repo": DATASET, "revision": IMAGE_REVISION, "tasks": tasks, "bundles": bundles, "files": files}


def test_both_kept_tables_bind_nested_sif_paths_without_collapsing_the_directory():
    index = native_image_index()
    files = image_files(index)
    assert len(files) == EXPECTED_IMAGES
    first = files["0" * 64]
    assert first["path"] == "cached_images/" + "0" * 64 + "/image-0.sif/image-0.sif"
    assert first["sha256"] == index["bundles"]["0" * 64]["files"]["image-0.sif"]["sha256"]
    # A v3-only image is required too; a v2-only index cannot authorize conversion.
    broken = copy.deepcopy(index)
    broken["tasks"]["endless_v3"].pop("task-0")
    with pytest.raises(ValueError, match="both kept populations"):
        image_files(broken)


def test_file_readback_cannot_change_the_pristine_bundle_bytes():
    index = native_image_index()
    first = next(iter(index["files"].values()))
    first["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="checksum binding differs"):
        image_files(index)
