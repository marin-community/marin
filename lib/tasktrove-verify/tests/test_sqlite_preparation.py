# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finite SQLite policies feed literal Exact without owning comparison."""

import pytest
from tasktrove_verify.modes.grade_exact import grade_exact_candidate
from tasktrove_verify.preparation.sqlite import prepare_sqlite_row_set
from tasktrove_verify.spec import EmptyOutputPolicy, ExactSpec


@pytest.mark.parametrize(
    "reference,candidate,reward",
    [
        (((1,), (1,), (2,)), ((2.0,), (1.0,)), 1),
        (((0,),), ((-0.0,),), 1),
        (((None,),), (("None",),), 0),
        (((b"abc",),), (("abc",),), 0),
        (((1, 2),), ((2, 1),), 0),
        (((9007199254740993,),), ((9007199254740992.0,),), 0),
        ((), (), 1),
        ((), ((None,),), 0),
        (((1 / 3,),), ((0.333333,),), 0),
    ],
)
def test_sqlite_row_set_policy_preserves_source_equivalence(reference, candidate, reward):
    spec = ExactSpec(
        expected=(prepare_sqlite_row_set(reference),),
        ignore_case=False,
        ignore_whitespace=False,
        strip_outer_whitespace=False,
        empty_output=EmptyOutputPolicy.GRADE,
    )
    assert grade_exact_candidate(spec, prepare_sqlite_row_set(candidate)).reward == reward


def test_nonfinite_observation_cannot_produce_an_exact_reference():
    with pytest.raises(ValueError, match="finite"):
        prepare_sqlite_row_set(((float("inf"),),))
