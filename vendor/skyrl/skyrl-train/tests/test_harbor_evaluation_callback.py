# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor evaluation callback configuration checks."""

import pytest

from skyrl_train.callbacks.builtin import HarborEvaluationCallback


def _cohort(environment):
    return {"name": "validation", "job": {"agents": [{}], "tasks": [], "environment": environment}}


@pytest.mark.parametrize("environment", [{"type": "daytona"}, {"type": "daytona", "kwargs": {"image_build": "snapshot"}}])
def test_harbor_evaluation_accepts_snapshot_image_builds(environment):
    HarborEvaluationCallback(
        cohorts=[_cohort(environment)], timeout=60, minimum_coverage=1.0, daytona_api_key_env="DAYTONA_EVAL_API_KEY"
    )


def test_harbor_evaluation_rejects_declarative_image_builds_at_construction():
    with pytest.raises(ValueError):
        HarborEvaluationCallback(
            cohorts=[_cohort({"type": "daytona", "kwargs": {"image_build": "declarative"}})],
            timeout=60,
            minimum_coverage=1.0,
            daytona_api_key_env="DAYTONA_EVAL_API_KEY",
        )
