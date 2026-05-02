# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.dev_tpu import DevTpuState, DevTpuWorker, GcpNodeRef, parse_worker_host


def test_parse_worker_host_accepts_http_address():
    assert parse_worker_host("http://10.0.0.12:10001") == "10.0.0.12"


def test_parse_worker_host_accepts_host_port_without_scheme():
    assert parse_worker_host("10.0.0.13:10001") == "10.0.0.13"


def test_parse_worker_host_rejects_missing_host():
    with pytest.raises(ValueError, match="host"):
        parse_worker_host("http://:10001")


def test_dev_tpu_state_json_roundtrip():
    state = DevTpuState(
        session_name="dlwh-branch-123456",
        config_file="/tmp/iris.yaml",
        job_id="/dlwh/dev-tpu-dlwh-branch-123456",
        tpu_type="v5p-8",
        workers=[
            DevTpuWorker(
                task_id="/dlwh/dev-tpu-dlwh-branch-123456/0",
                worker_id="10.0.0.12",
                worker_address="http://10.0.0.12:10001",
                host="10.0.0.12",
                node=GcpNodeRef(
                    kind="tpu",
                    name="iris-dev-tpu",
                    zone="us-east5-a",
                    project="hai-gcp-models",
                    tpu_worker_id=0,
                ),
            )
        ],
    )

    restored = DevTpuState.from_json(state.to_json())

    assert restored == state
