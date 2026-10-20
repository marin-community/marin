# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.mlx_export.provenance import Recipe, file_digest, validate_dump, verified_build, write_manifest


@pytest.mark.parametrize("changed", ["weights", "recipe", "golden", "source"])
def test_parity_rejects_a_dump_from_different_inputs(tmp_path, changed):
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"original model weights")
    revision = "a" * 40
    write_manifest(model, repo="fixture/model", revision=revision, recipe=Recipe.MIXED_4BIT)
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{"source_repo": "fixture/model", "source_revision": revision}]))
    dump = {"build": verified_build(model), "golden_sha256": file_digest(golden)}
    assert validate_dump(dump, model=model, recipe=Recipe.MIXED_4BIT, golden=golden) == dump["build"]

    recipe = Recipe.MIXED_4BIT
    if changed == "weights":
        (model / "weights.safetensors").write_bytes(b"different model weights")
    elif changed == "recipe":
        recipe = Recipe.EIGHT_BIT
    elif changed == "golden":
        golden.write_text(golden.read_text() + "\n")
    else:
        golden.write_text(json.dumps([{"source_repo": "different/model", "source_revision": revision}]))
        dump["golden_sha256"] = file_digest(golden)
    with pytest.raises(ValueError):
        validate_dump(dump, model=model, recipe=recipe, golden=golden)
