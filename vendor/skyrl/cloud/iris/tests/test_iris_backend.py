# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.types import JobName

from cloud.iris.iris_backend import launch


@pytest.mark.parametrize(
    "routing", 
    [
        {"target_cluster": "cw-rno2a", "parent_cluster_config": "marin.yaml"},
        {"target_cluster": None, "parent_cluster_config": "marin.yaml"},
    ],
)
def test_nested_launch_rejects_root_routing_before_remote_submission(tmp_path, routing):
    previous = get_job_info()
    set_job_info(JobInfo(task_id=JobName.from_wire("/alice/coordinator/0")))
    try:
        with pytest.raises(ValueError, match="Clear target_cluster and parent_cluster_config"):
            launch(SimpleNamespace(**routing), tmp_path)
    finally:
        set_job_info(previous)
