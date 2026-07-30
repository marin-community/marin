# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.grug.moe.eval_augdk_generation import _eligibility


def test_shared_expert_selection_broadcasts_over_batch() -> None:
    eligibility = _eligibility(3, 8, ((0, 2, 7),))

    assert eligibility is not None
    np.testing.assert_array_equal(
        eligibility,
        np.asarray(
            [
                [True, False, True, False, False, False, False, True],
                [True, False, True, False, False, False, False, True],
                [True, False, True, False, False, False, False, True],
            ]
        ),
    )


def test_layerwise_expert_selection_preserves_each_layer_mask() -> None:
    eligibility = _eligibility(2, 4, ((0, 1), (2, 3)))

    assert eligibility is not None
    np.testing.assert_array_equal(
        eligibility,
        np.asarray(
            [
                [[True, True, False, False], [True, True, False, False]],
                [[False, False, True, True], [False, False, True, True]],
            ]
        ),
    )
