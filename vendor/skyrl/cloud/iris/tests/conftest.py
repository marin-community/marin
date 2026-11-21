# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

from cloud.iris.ray_ports import RayPort


@pytest.fixture
def iris_ray_ports(monkeypatch):
    ports = {port: 12000 + index for index, port in enumerate(RayPort)}
    for port, value in ports.items():
        monkeypatch.setenv(f"IRIS_PORT_{port.value.upper()}", str(value))
    return ports


@pytest.fixture
def worker_config():
    return OmegaConf.create(
        {
            "backend": "vllm",
            "max_input_length": 3,
            "max_model_len": 128,
            "apply_overlong_filtering": False,
            "sampling_params": {
                "max_generate_length": 8,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0,
                "logprobs": 0,
            },
        }
    )
