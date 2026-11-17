# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verified SWE conversion binds source images and the protected grader together."""

import json
import zlib

import pytest
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.review import review_payload
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.prime_intellect import swe


@pytest.mark.parametrize("family", ["scale", "lego"])
def test_conversion_uses_frozen_image_and_preserves_private_source_scorer(family, tmp_path):
    source = "PrimeIntellect/" + ("Scale-SWE-Verified" if family == "scale" else "SWE-Lego-Real-Data-Verified")
    original = "example.test/source:mutable"
    pinned = "example.test/source@sha256:" + "b" * 64
    data = {
        "instance_id": "fixture",
        "problem_statement": "Repair the calculation without changing the public API.",
        "patch": "--- a/module.py\n+++ b/module.py\n@@ -1 +1 @@\n-return 0\n+return 1\n",
        "image_url": original,
        "image_name": original,
        "workdir": "/source-repo",
        "pre_commands": "true",
        "parent_commit": "a" * 40,
        "base_commit": "a" * 40,
        "f2p_patch": "",
        "f2p_script": "",
        "test_patch": "",
        "test_cmd": "python -m pytest -rA tests/test_fix.py",
        "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
        "PASS_TO_PASS": [],
    }
    manifest = {
        "dataset_id": source,
        "revision": "pin",
        "failures": [],
        "source_tasks": 1,
        "resolved": [
            {
                "original_reference": original,
                "pinned_reference": pinned,
                "source_rows": [{"instance_id": "fixture", "source_row": "data/train.parquet:0"}],
            }
        ],
    }
    (tmp_path / swe.IMAGE_BINDINGS_FILENAME).write_text(json.dumps(manifest))
    row = RawRow(
        "fixture",
        Source(dataset=source, revision="pin", row="data/train.parquet:0", importer_revision=swe.VERSION),
        data,
    )
    context = ConversionContext({swe.IMAGE_BINDINGS_INPUT: StoragePath(str(tmp_path))}, None)
    convert = swe.convert_scale if family == "scale" else swe.convert_lego
    task = convert(row, context)
    assert isinstance(task, TaskSpec)
    assert task.integrity_contract is not None
    assert task.integrity_contract.workspace == ("/source-repo" if family == "scale" else "/testbed")
    files = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    assert files["environment/Dockerfile"].decode().startswith("FROM " + pinned + "\n")
    assert not {"tests/config.json", "tests/native_scripts.py", "tests/grade.py"} & files.keys()
    private = json.loads(zlib.decompress(files["setup_files/taskcompendium-source-input.zlib"]))
    assert bytes.fromhex(private["tests"]["native_scripts.py"]) == (swe.HERE / "native_scripts.py").read_bytes()
    config = json.loads(bytes.fromhex(private["tests"]["config.json"]))
    assert config["test_ids"] == ["tests/test_fix.py::test_fix"]
    original_task = task.model_dump_json()
    projected = review_payload(task)
    assert projected == review_payload(task, payload_revision="5")
    grader_text = {
        resource["path"]: resource["text"]
        for resource in projected["resources"]
        if resource["role"] == "source_verifier"
    }
    grader = "scale_grade.py" if family == "scale" else "lego_grade.py"
    assert grader_text["tests/grade.py"] == (swe.HERE / grader).read_text()
    assert grader_text["tests/native_scripts.py"] == (swe.HERE / "native_scripts.py").read_text()
    assert "tests/test_fix.py::test_fix" in grader_text["tests/config.json"]
    historical = review_payload(task, payload_revision="4")
    assert not any(resource["role"] == "source_verifier" for resource in historical["resources"])
    assert task.model_dump_json() == original_task
    assert private["spec"]["timeout"] is None
    assert task.context.events[0].content.endswith(swe.SOLVER_SYSTEM_PROMPT + "\n\n" + data["problem_statement"])
    solution = {resource.path: resource_bytes(resource) for resource in task.resources.oracle}
    assert solution["solution/gold.patch"] == data["patch"].encode()
    assert task.integrity_contract.id == "shared-source"
