# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from mariner.data import PAPER_TARGET_BUDGET, SWARMS, WEIGHT_PREFIX, read_table, release_partition
from mariner.design import proportional, removals, stratified, unimax

BASELINE_TOLERANCE = 1e-15  # agreed: the designed baselines against the released rows; the largest gap is 5.8e-16
UNIMAX_BUDGET_FRACTION = 0.8
UNIMAX_MAX_EPOCHS = 1.0


@pytest.mark.data_integration
@pytest.mark.timeout(60)
@pytest.mark.parametrize("swarm", SWARMS)
def test_design_baselines_equal_released_swarm_rows(snapshot, swarm):
    partition = release_partition(snapshot)
    runs = read_table(snapshot, f"swarm_{swarm}")
    weight_columns = [f"{WEIGHT_PREFIX}{bucket}" for bucket in partition.bucket_ids]
    proportional_weights = proportional(partition.pool_tokens)
    designed = {
        "baseline_proportional": proportional_weights,
        "baseline_unimax": unimax(
            partition.pool_tokens, UNIMAX_BUDGET_FRACTION * PAPER_TARGET_BUDGET, UNIMAX_MAX_EPOCHS
        ),
        "baseline_stratified": stratified(len(partition.bucket_ids)),
    }
    released = runs.set_index("run_label")[weight_columns]
    for label, weights in designed.items():
        np.testing.assert_allclose(released.loc[label].to_numpy(float), weights, rtol=0, atol=BASELINE_TOLERANCE)

    # Each deletion row zeroes one bucket, and the 39 rows zero the 39 buckets.
    deletions = runs.loc[runs["design_kind"] == "deletion", weight_columns].to_numpy(float)
    assert ((deletions == 0).sum(axis=1) == 1).all()
    zeroed = (deletions == 0).argmax(axis=1)
    assert sorted(zeroed) == list(range(len(partition.bucket_ids)))
    np.testing.assert_allclose(deletions, removals(proportional_weights)[zeroed], rtol=0, atol=BASELINE_TOLERANCE)
