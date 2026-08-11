# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live-process coverage for the controller rollout smoke suite."""

import pytest
from iris.client import LocalClientConfig
from iris.client.local_client import local_client

from scripts.iris.rollout_controllers import run_smoke_suite


@pytest.mark.requires_cluster
def test_rollout_smoke_suite_exercises_resource_lifecycle_on_local_cluster():
    with local_client(LocalClientConfig(max_workers=1)) as client:
        result = run_smoke_suite(client, timeout=120)

    assert len({result.completed_job, result.cancelled_job, result.followup_job}) == 3
    assert result.cancel_action_id
