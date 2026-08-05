# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np

from scripts.perf.grug_fixed_replay import (
    build_loss_weight,
    compare_sampled_action_log_probs,
    repacked_operational_micro_loss,
    representative_action_coordinates,
)
from scripts.perf.grug_levanter_fixed_replay_benchmark import (
    reconstruct_reference_coordinates,
    tree_finite_evidence,
)


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


def test_representative_action_coordinates_match_msrl_causal_positions():
    assert representative_action_coordinates(sequence_length=8, action_length=4, microbatch_count=2) == [
        {
            "microbatch": 0,
            "representative_position": "first",
            "worker_sample_index": 0,
            "action_index": 0,
            "model_token_index": 3,
        },
        {
            "microbatch": 0,
            "representative_position": "middle",
            "worker_sample_index": 1,
            "action_index": 2,
            "model_token_index": 5,
        },
        {
            "microbatch": 0,
            "representative_position": "last",
            "worker_sample_index": 2,
            "action_index": 3,
            "model_token_index": 6,
        },
        {
            "microbatch": 1,
            "representative_position": "first",
            "worker_sample_index": 3,
            "action_index": 0,
            "model_token_index": 3,
        },
        {
            "microbatch": 1,
            "representative_position": "middle",
            "worker_sample_index": 4,
            "action_index": 2,
            "model_token_index": 5,
        },
        {
            "microbatch": 1,
            "representative_position": "last",
            "worker_sample_index": 5,
            "action_index": 3,
            "model_token_index": 6,
        },
    ]


def test_reconstruct_reference_coordinates_uses_frozen_msrl_sample_order():
    workers = [{"rank": 3, "microbatches": 2, "representative_action_log_probs": list(range(6))}]

    reconstructed = reconstruct_reference_coordinates(workers, sequence_length=8, action_length=4)

    assert reconstructed[0]["representative_action_log_prob_coordinates"] == representative_action_coordinates(
        sequence_length=8, action_length=4, microbatch_count=2
    )
    assert "representative_action_log_prob_coordinates" not in workers[0]


def test_compare_sampled_action_log_probs_reports_descriptive_pair_preference():
    coordinates = representative_action_coordinates(sequence_length=8, action_length=4, microbatch_count=1)

    def worker(values):
        return {
            "rank": 0,
            "representative_action_log_probs": values,
            "representative_action_log_prob_coordinates": coordinates,
        }

    result = compare_sampled_action_log_probs(
        [worker([-1.0, -2.0, -3.0])],
        {
            "eager": [worker([-1.2, -1.8, -3.0])],
            "grouped": [worker([-1.1, -2.4, -3.0])],
        },
    )

    assert result["distances_from_levanter"]["eager"]["checked"] == 3
    np.testing.assert_allclose(result["distances_from_levanter"]["eager"]["mean_abs_difference"], 0.4 / 3)
    assert result["distances_from_levanter"]["grouped"]["max_abs_difference_sample"]["coordinate"] == {
        "rank": 0,
        **coordinates[1],
    }
    pair = result["paired_preference"]
    assert {
        name: pair[name]
        for name in (
            "left_arm",
            "right_arm",
            "checked",
            "changed",
            "unchanged",
            "left_closer_on_changed",
            "right_closer_on_changed",
            "ties_on_changed",
        )
    } == {
        "left_arm": "eager",
        "right_arm": "grouped",
        "checked": 3,
        "changed": 2,
        "unchanged": 1,
        "left_closer_on_changed": 1,
        "right_closer_on_changed": 1,
        "ties_on_changed": 0,
    }
    np.testing.assert_allclose(pair["mean_abs_error_delta_right_minus_left"], 0.1 / 3)
    np.testing.assert_allclose(pair["sum_squared_error_left"], 0.08)
    np.testing.assert_allclose(pair["sum_squared_error_right"], 0.17)
    np.testing.assert_allclose(pair["sum_squared_error_right_over_left"], 2.125)
    assert pair["right_closer_fraction_of_changed"] == 0.5


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
