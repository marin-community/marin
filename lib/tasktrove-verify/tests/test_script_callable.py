# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from tasktrove_verify.grade import Status
from tasktrove_verify.modes.grade_script import grade_script_callable

from .bounded_grading import trusted_script

pytestmark = pytest.mark.usefixtures("importable_grading_modules")


@pytest.mark.parametrize(
    "behavior,status,reward",
    [
        ("success", Status.SCORED, 0.25),
        ("invalid_reference", Status.INVALID_TASK, 0),
        ("runtime_failure", Status.INFRA_ERROR, 0),
        ("unscored_positive", Status.INFRA_ERROR, 0),
        ("nonfinite", Status.INFRA_ERROR, 0),
        ("malformed", Status.INFRA_ERROR, 0),
        ("invalid_detail", Status.INFRA_ERROR, 0),
        ("timeout", Status.INFRA_ERROR, 0),
    ],
)
def test_script_callable_preserves_status_and_rejects_unusable_rewards(behavior, status, reward):
    verdict = grade_script_callable(trusted_script, behavior, timeout=0.3 if behavior == "timeout" else 5)
    assert (verdict.status, verdict.reward) == (status, reward)
    if behavior == "success":
        assert verdict.detail == {"observations": [True, False, False, False]}
