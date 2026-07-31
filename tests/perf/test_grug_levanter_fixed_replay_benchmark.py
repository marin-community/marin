# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from scripts.perf.grug_fixed_replay import build_loss_weight


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
