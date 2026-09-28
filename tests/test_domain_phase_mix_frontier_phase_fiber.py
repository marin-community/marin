# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib

import numpy as np

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as augmented
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_3e18_frontier_phase_fiber_panel as fiber,
)


def test_frontier_phase_fiber_panel_preserves_aggregate_and_pairing(tmp_path) -> None:
    domains = tuple(augmented.DOMAIN_NAMES)
    alpha0, alpha1 = fiber._realized_phase_fractions()
    anchors, _anchor_audit = fiber.load_anchors(domains)
    candidates = fiber.build_candidates(anchors, domains, alpha0, alpha1)
    manifest, weights, summary = fiber.validate_candidates(candidates, anchors, domains, alpha0, alpha1)

    assert len(manifest) == 200
    assert {anchor.uncheatable_one_phase_rank for anchor in anchors if anchor.anchor_id == "uncheatable_frontier"} == {
        1.0
    }
    assert {anchor.table9_one_phase_rank for anchor in anchors if anchor.anchor_id == "table9_frontier"} == {1.0}
    assert manifest.groupby(["anchor_id", "seed_block"]).size().eq(25).all()
    assert summary["direction_rank"] == {anchor.anchor_id: 38 for anchor in anchors}
    assert max(summary["normalized_direction_condition_number"].values()) < 2

    anchor_lookup = {anchor.anchor_id: anchor.weights for anchor in anchors}
    aggregates = alpha0 * weights[:, 0] + alpha1 * weights[:, 1]
    expected_aggregates = np.stack([anchor_lookup[anchor_id] for anchor_id in manifest["anchor_id"]])
    np.testing.assert_allclose(aggregates, expected_aggregates, atol=2e-12, rtol=0)

    for (anchor_id, _direction_id), pair in manifest.loc[~manifest["sign"].eq("center")].groupby(
        ["anchor_id", "direction_id"]
    ):
        assert set(pair["sign"]) == {"plus", "minus"}
        assert pair["data_seed"].nunique() == 1
        pair_midpoint = weights[pair.index.to_numpy(dtype=int)].mean(axis=0)
        expected_midpoint = np.stack([anchor_lookup[anchor_id], anchor_lookup[anchor_id]])
        np.testing.assert_allclose(pair_midpoint, expected_midpoint, atol=2e-12, rtol=0)

    source_path, source_sha256 = fiber.write_launcher_source_panel(tmp_path, manifest, weights, domains)
    assert source_sha256 == "8d2c102955149265f2f187c35b899cd250aab618d1176758799f37f9b4f146e8"
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_sha256
