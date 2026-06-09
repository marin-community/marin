# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from scripts.analysis.build_decon_val_sweep import cutoff_tag, derive_cutoff_keepsets


def test_cutoff_tag_zero_pads() -> None:
    assert cutoff_tag(0.5) == "j050"
    assert cutoff_tag(0.55) == "j055"
    assert cutoff_tag(0.9) == "j090"


def test_derive_cutoff_keepsets_thresholds_nests_and_counts_tokens() -> None:
    ids = ["a", "b", "c", "d", "e"]
    doc_indices = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    # a, e clean; b/c/d contaminated at ascending Jaccard. d (0.95) is never kept.
    aligned_max_jaccard = np.array([0.0, 0.6, 0.8, 0.95, 0.0], dtype=np.float64)
    doc_tokens = np.zeros(15, dtype=np.int64)
    doc_tokens[[10, 11, 12, 13, 14]] = [100, 200, 300, 400, 500]

    out = derive_cutoff_keepsets(ids, doc_indices, aligned_max_jaccard, doc_tokens, [0.5, 0.75, 0.9])

    assert out[0.5]["ids"] == ["a", "e"]
    assert out[0.75]["ids"] == ["a", "b", "e"]
    assert out[0.9]["ids"] == ["a", "b", "c", "e"]
    assert out[0.5]["doc_indices"].tolist() == [10, 14]
    # token sums add the newly-admitted doc at each looser cutoff
    assert out[0.5]["tokens"] == 600
    assert out[0.75]["tokens"] == 800
    assert out[0.9]["tokens"] == 1100
    # strict nesting across ascending cutoffs
    assert set(out[0.5]["ids"]) < set(out[0.75]["ids"]) < set(out[0.9]["ids"])
