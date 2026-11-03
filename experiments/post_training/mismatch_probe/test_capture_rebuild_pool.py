# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The pool-rebuild capture's dispatch plan."""

import pytest

from experiments.post_training.mismatch_probe.capture_rebuild_pool import slabs


def test_the_boundary_set_splits_across_pools_and_later_sets_follow_the_fresh_one():
    sets = [("P_X", 18), ("P_R", 50), ("P_chat", 29), ("P_think", 8)]
    plan = slabs(sets, ("P_R", 40))
    assert [(pool, name, start, end) for pool, name, start, end in plan] == [
        (0, "P_X", 0, 18),
        (0, "P_R", 0, 40),
        (1, "P_R", 40, 50),
        (1, "P_chat", 0, 29),
        (1, "P_think", 0, 8),
    ]


def test_a_leading_boundary_split_is_the_rebuild_boundary():
    sets = [("P_X", 18)]
    with pytest.raises(ValueError, match="interior"):
        slabs(sets, ("P_X", 0))
    plan = slabs(sets, ("P_X", 1))
    assert plan == [(0, "P_X", 0, 1), (1, "P_X", 1, 18)]


def test_a_trailing_boundary_group_is_the_fresh_pool_alone():
    plan = slabs([("P_R", 50)], ("P_R", 49))
    assert plan == [(0, "P_R", 0, 49), (1, "P_R", 49, 50)]


def test_an_unknown_boundary_set_is_rejected():
    with pytest.raises(ValueError, match="interior"):
        slabs([("P_X", 18), ("P_R", 50)], ("P_chat", 3))
