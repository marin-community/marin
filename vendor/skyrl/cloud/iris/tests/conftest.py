# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from cloud.iris.ray_ports import RayPort


@pytest.fixture
def iris_ray_ports(monkeypatch):
    ports = {port: 12000 + index for index, port in enumerate(RayPort)}
    for port, value in ports.items():
        monkeypatch.setenv(f"IRIS_PORT_{port.value.upper()}", str(value))
    return ports
