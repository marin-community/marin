# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import draccus
import numpy as np
import pytest
from levanter.models.snowball import SnowballConfig
from safetensors import safe_open

from experiments.benchmarks.diagnose_embedding_generation import snowball_config_from_report
from experiments.benchmarks.matched_inference import NormWeights, export_fixture


@pytest.mark.parametrize("recipe", ["snowball", "hero"])
@pytest.mark.timeout(120)
def test_nonunit_fixture_changes_only_norm_weights_and_roundtrips(tmp_path, recipe):
    original, nonunit = tmp_path / "ones", tmp_path / "nonunit"
    # Each export reloads the actual HF checkpoint and checks every tensor exactly.
    export_fixture(original, recipe)
    export_fixture(nonunit, recipe, NormWeights.NONUNIT)
    changed = set()
    with (
        safe_open(original / "checkpoint/model.safetensors", framework="np") as left,
        safe_open(nonunit / "checkpoint/model.safetensors", framework="np") as right,
    ):
        assert left.keys() == right.keys()
        for name in left.keys():
            before, after = left.get_tensor(name), right.get_tensor(name)
            if name.endswith("norm.weight"):
                np.testing.assert_array_equal(before, np.ones_like(before))
                assert np.any(after != before), name
                changed.add(name)
            else:
                np.testing.assert_array_equal(before, after, err_msg=name)
    old_manifest = json.loads((original / "manifest.json").read_text())
    manifest = json.loads((nonunit / "manifest.json").read_text())
    assert changed
    assert changed == set(manifest["norm_weights"]["modified_tensors"])
    assert old_manifest["checkpoint"] != manifest["checkpoint"]
    assert json.loads((original / "workload.json").read_text()) == json.loads((nonunit / "workload.json").read_text())


def test_native_report_concrete_config_restores_diagnostic_recipe():
    # Match the native driver's draccus.encode(concrete_config) wire format:
    # it carries no "type" discriminator, unlike encoding through LmConfig.
    config = SnowballConfig(num_layers=2, hidden_dim=256, vocab_size=256, inference_attention_implementation="reference")
    provenance = {
        "model_config": draccus.encode(config),
        "checkpoint": {"hf_config": config.to_hf_config(config.vocab_size).to_dict()},
    }
    restored = snowball_config_from_report(json.loads(json.dumps(provenance)))
    assert restored == config
