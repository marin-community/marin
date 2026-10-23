# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import math
import sys

import pytest
import torch
from safetensors.torch import save_file

from experiments.weight_merging.playground.distances import main


@pytest.mark.parametrize("sharded", [False, True])
def test_checkpoint_diagnostic_measures_small_updates_lost_by_bf16_rounding(tmp_path, monkeypatch, sharded):
    anchor, donor = tmp_path / "anchor", tmp_path / "donor"
    anchor.mkdir()
    donor.mkdir()
    values = {"weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16), "norm": torch.tensor([1.0])}
    updates = {"weight": torch.tensor([1.0001, 2.02], dtype=torch.float32), "norm": torch.tensor([1.0])}
    for path, tensors in ((anchor, values), (donor, updates)):
        if sharded:
            mapping = {}
            for index, (name, tensor) in enumerate(tensors.items()):
                filename = f"shard-{index}.safetensors"
                save_file({name: tensor}, path / filename)
                mapping[name] = filename
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
        else:
            save_file(tensors, path / "model.safetensors")
    output = tmp_path / "distance.json"
    monkeypatch.setattr(
        sys, "argv", ["distances", "--anchor", str(anchor), "--donor", str(donor), "--output", str(output)]
    )
    main()
    actual = json.loads(output.read_text())
    assert actual["elements"] == 3
    assert actual["active"] == 2
    assert actual["active_rounded_to_zero"] == 1
    differences = [float(updates["weight"][0]) - 1, float(updates["weight"][1]) - 2]
    expected_distance = math.sqrt(sum(value * value for value in differences) / 6)
    assert actual["relative_l2"] == pytest.approx(expected_distance, rel=1e-12)
    assert actual["rounding_error_over_delta"] > 0.2
    for filename, digest in actual["donor_sha256"].items():
        assert digest == hashlib.sha256((donor / filename).read_bytes()).hexdigest()
