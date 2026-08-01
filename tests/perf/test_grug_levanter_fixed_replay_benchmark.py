# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np

from scripts.perf.grug_fixed_replay import build_loss_weight, repacked_operational_micro_loss
from scripts.perf.grug_levanter_fixed_replay_benchmark import tree_finite_evidence


def test_build_loss_weight_matches_skyrl_action_logprob_slice():
    loss_mask = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]], dtype=np.float32)

    result = build_loss_weight(loss_mask, sequence_length=6)

    np.testing.assert_array_equal(
        result,
        np.asarray(
            [
                [0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
                [0.0, 0.0, 4.0, 5.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_build_loss_weight_rejects_actions_longer_than_next_token_positions():
    with np.testing.assert_raises(ValueError):
        build_loss_weight(np.ones((1, 4), dtype=np.float32), sequence_length=4)


def test_repacked_operational_loss_uses_token_sum_and_router_mean():
    ce_sums = np.asarray([10.0, 30.0])
    router_aux_losses = np.asarray([2.0, 6.0])

    result = sum(
        repacked_operational_micro_loss(
            ce_sum,
            router_aux,
            global_loss_tokens=40,
            microbatch_count=2,
        )
        for ce_sum, router_aux in zip(ce_sums, router_aux_losses, strict=True)
    )

    assert result == 5.0


def test_tree_finite_evidence_preserves_paths_and_nonfinite_counts():
    tree = {
        "finite": jnp.asarray([1.0, -3.0], dtype=jnp.float32),
        "nested": {"bad": jnp.asarray([jnp.nan, jnp.inf, 2.0], dtype=jnp.float32)},
        "ignored": None,
    }

    evidence = tree_finite_evidence(tree)

    assert evidence["checked_arrays"] == 2
    assert evidence["checked_elements"] == 5
    assert evidence["nonfinite_arrays"] == 1
    assert evidence["nonfinite_elements"] == 2
    assert evidence["max_finite_abs"] == 3.0
    assert evidence["leaves"] == [
        {
            "path": "['finite']",
            "shape": [2],
            "dtype": "float32",
            "elements": 2,
            "finite": True,
            "nonfinite_elements": 0,
            "max_finite_abs": 3.0,
        },
        {
            "path": "['nested']['bad']",
            "shape": [3],
            "dtype": "float32",
            "elements": 3,
            "finite": False,
            "nonfinite_elements": 2,
            "max_finite_abs": 2.0,
        },
    ]
