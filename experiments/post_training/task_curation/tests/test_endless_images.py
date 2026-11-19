# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import sys

import pytest

from experiments.post_training.task_curation.datasets.endless import images as endless_images
from experiments.post_training.task_curation.datasets.endless.images import (
    DATASET,
    EXPECTED_IMAGES,
    EXPECTED_TASKS,
    IMAGE_REVISION,
    image_files,
)
from experiments.post_training.task_curation.datasets.facet import images as facet_images


def native_image_index(image_count=EXPECTED_IMAGES, task_counts=EXPECTED_TASKS):
    bundles = {}
    files = {}
    bundle_ids = []
    for index in range(image_count):
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
    for folder, count in task_counts.items():
        tasks[folder] = {
            f"task-{row}": {"environments": [{"bundle": bundle_ids[(offset + row) % image_count]}]}
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


@pytest.mark.parametrize("producer,namespace", [(facet_images, "facet"), (endless_images, "endless")])
def test_shared_cli_preserves_each_sources_artifact_namespace(producer, namespace, tmp_path, monkeypatch, capsys):
    index = native_image_index(producer.EXPECTED_IMAGES, producer.EXPECTED_TASKS)
    index["revision"] = producer.IMAGE_REVISION
    data = json.dumps(index).encode()
    index_path = tmp_path / "index.json"
    index_path.write_bytes(data)
    tools = tmp_path / "tools.json"
    tools.write_text("{}")
    bundle = next(iter(index["bundles"]))
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "images",
            "--index",
            str(index_path),
            "--index-sha256",
            hashlib.sha256(data).hexdigest(),
            "--licenses",
            str(tmp_path / "licenses.json"),
            "--licenses-sha256",
            "a" * 64,
            "--tools",
            str(tools),
            "--tools-sha256",
            hashlib.sha256(tools.read_bytes()).hexdigest(),
            "--workdir",
            str(tmp_path / "work"),
            "--artifact-root",
            "s3://marin-us-east-02a/marin/image-test",
            "--artifact-version",
            "2026.11.19.999",
            "--max-concurrent",
            "1",
            "--bundle",
            bundle,
            "--output",
            str(output),
        ],
    )
    producer.main()
    plan = json.loads(capsys.readouterr().out)
    assert len(plan) == 1
    assert f"/rl-data/{namespace}/pristine-oci/{bundle}/2026.11.19.999" in plan[0]["path"]
    assert not output.exists()
