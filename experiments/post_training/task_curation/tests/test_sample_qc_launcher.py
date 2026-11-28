# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The SAMPLE QC launcher refuses GPU flavors its target cluster cannot place."""

import pytest

from experiments.post_training.task_curation.judges import require_review_serving_root


@pytest.mark.parametrize("cluster", ["cw-rno2a", "cw-us-east-02a", None])
def test_refuses_clusters_that_place_no_gb200(cluster):
    with pytest.raises(ValueError, match="4xGB200"):
        require_review_serving_root(cluster)


def test_accepts_a_gb200_cluster():
    assert require_review_serving_root("cw-us-east-08a") == "cw-us-east-08a"
