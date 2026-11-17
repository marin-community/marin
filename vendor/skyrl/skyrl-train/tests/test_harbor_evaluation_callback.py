# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from skyrl_train.callbacks.builtin import HarborEvaluationCallback


def test_harbor_evaluation_rejects_invalid_image_build_at_construction():
    with pytest.raises(ValueError):
        HarborEvaluationCallback(
            cohorts=[{
                "name": "validation",
                "job": {
                    "agents": [{}],
                    "tasks": [],
                    "environment": {"type": "daytona", "kwargs": {"image_build": "invalid"}},
                },
            }],
            timeout=60,
            minimum_coverage=1.0,
            minimum_native_coverage=1.0,
            daytona_api_key_env="DAYTONA_EVAL_API_KEY",
        )
