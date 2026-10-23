# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

from cloud.iris.ray_ports import RayPort

# Inventory producer tests import the maintained experiment from the outer workspace.
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))


@pytest.fixture
def iris_ray_ports(monkeypatch):
    ports = {port: 12000 + index for index, port in enumerate(RayPort)}
    for port, value in ports.items():
        monkeypatch.setenv(f"IRIS_PORT_{port.value.upper()}", str(value))
    return ports
