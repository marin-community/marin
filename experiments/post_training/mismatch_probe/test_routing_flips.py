# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from experiments.post_training.mismatch_probe.routing_flips import compare_routes


def test_routing_flip_report_counts_sets_at_aligned_token_layer_pairs(tmp_path):
    selected = np.array([[[1, 2], [3, 4]]] * 4)
    changed = selected[..., ::-1].copy()
    changed[1, 0] = [2, 5]
    changed[3, 1] = [4, 5]
    for name, ids in (("base", selected), ("candidate", changed)):
        folder = tmp_path / name
        folder.mkdir()
        np.savez(folder / "s.npz", selected=ids, layers=[2, 7], margin=np.repeat(np.arange(4)[:, None], 2, axis=1))
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "model_id": name,
                    "complete": True,
                    "records": [
                        {
                            "file": "s.npz",
                            "sample_id": "s",
                            "probe_hash": "same-frozen-tokens",
                            "tokens": 4,
                            "token_type_tags": ["context", "context", "answer", "answer"],
                            "position_tags": [0, 0, 1, 1],
                        }
                    ],
                }
            )
        )
    result = compare_routes(tmp_path / "base", tmp_path / "candidate")
    assert result["pairs"] == 8
    assert result["flips"] == 2
    assert result["flip_rate"] == pytest.approx(0.25)
    assert result["by_layer"]["2"]["flips"] == 1
    assert result["by_layer"]["7"]["flips"] == 1
    assert [result["base_margin_quartiles"][str(q)]["flips"] for q in range(4)] == [0, 1, 0, 1]
    assert result["buckets"]["context"]["flip_rate"] == pytest.approx(0.25)
    assert result["position_buckets"]["1"]["pairs"] == 4
    assert compare_routes(tmp_path / "base", tmp_path / "base")["flips"] == 0

    path = tmp_path / "candidate" / "manifest.json"
    mismatched = json.loads(path.read_text())
    mismatched["records"][0]["probe_hash"] = "different-token-sequence"
    path.write_text(json.dumps(mismatched))
    with pytest.raises(ValueError, match="probe_hash differs"):
        compare_routes(tmp_path / "base", tmp_path / "candidate")
