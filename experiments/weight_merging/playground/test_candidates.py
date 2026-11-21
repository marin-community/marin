# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import torch
from safetensors.torch import save_file

from experiments.weight_merging.playground.candidates import merged_weights, read_candidates


def test_checkpoint_streaming_distinguishes_donor_and_final_output_rounding(tmp_path):
    paths = {}
    for name, weights in (("anchor", [1.0, 1.0]), ("donor", [1.007, 1.013])):
        paths[name] = tmp_path / name
        paths[name].mkdir()
        save_file({"weight": torch.tensor(weights), "count": torch.tensor(7)}, paths[name] / "model.safetensors")
    recipe = {"method": "average", "coefficients": [0.5]}
    candidates = [
        {"name": "full", "source": "anchor", "donors": ["donor"], "recipe": recipe},
        {
            "name": "rounded_output",
            "source": "anchor",
            "donors": ["donor"],
            "recipe": recipe,
            "output_precision": "bfloat16_rounded",
        },
        {
            "name": "rounded_donor",
            "source": "anchor",
            "donors": ["donor"],
            "recipe": recipe,
            "donor_precision": "bfloat16_rounded",
        },
    ]
    manifest = tmp_path / "candidates.json"
    manifest.write_text(json.dumps(candidates))
    actual = {candidate.name: dict(merged_weights(paths, candidate)) for candidate in read_candidates(manifest)}
    torch.testing.assert_close(actual["full"]["weight"], torch.tensor([1.0035, 1.0065]), rtol=0, atol=1e-7)
    torch.testing.assert_close(actual["rounded_output"]["weight"], torch.tensor([1.0, 1.0078125]), rtol=0, atol=0)
    torch.testing.assert_close(actual["rounded_donor"]["weight"], torch.tensor([1.00390625, 1.0078125]), rtol=0, atol=0)
    assert actual["rounded_output"]["weight"].dtype == torch.float32
    for tensors in actual.values():
        assert tensors["count"].dtype == torch.int64 and tensors["count"].item() == 7
