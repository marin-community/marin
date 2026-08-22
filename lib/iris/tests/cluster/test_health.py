# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from rigging.timing import Duration

from iris.cluster.health import TaskHealthCheck, publish_task_health, task_health_port


def test_task_health_check_requires_kubernetes_compatible_durations():
    with pytest.raises(ValueError, match="whole seconds"):
        TaskHealthCheck(
            startup_timeout=Duration.from_seconds(30),
            period=Duration.from_ms(1500),
            request_timeout=Duration.from_seconds(1),
            failure_threshold=3,
        )
    with pytest.raises(ValueError, match="less than period"):
        TaskHealthCheck(
            startup_timeout=Duration.from_seconds(30),
            period=Duration.from_seconds(5),
            request_timeout=Duration.from_seconds(5),
            failure_threshold=3,
        )


def test_publish_task_health_uses_the_backend_selected_port(monkeypatch, tmp_path):
    port_file = tmp_path / "health-port"
    monkeypatch.setenv("IRIS_HEALTH_PORT_FILE", str(port_file))
    monkeypatch.setenv("IRIS_PORT_HEALTHZ", "43210")

    assert task_health_port() == 43210
    publish_task_health(43210)
    assert port_file.read_text() == "43210\n"

    with pytest.raises(ValueError, match="expected 43210"):
        publish_task_health(43211)
