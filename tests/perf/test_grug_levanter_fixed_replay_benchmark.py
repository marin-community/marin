# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from scripts.perf.grug_fixed_replay import build_loss_weight, repacked_operational_micro_loss


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
