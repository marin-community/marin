"""Simple test that just checks whether scheduling a task on
a GPU works.
"""

import os

import pytest
import ray

from tests.conftest import SINGLE_GPU_CONFIG


@ray.remote
def increment(x):
    import torch
    torch.cuda.device_count()

    return x + 1

@pytest.mark.skipif(os.getenv("GPU_CI") != "true", reason="Skip this test if not running with a GPU in CI.")
def test_scheduling_remote_func_on_gpu(ray_cluster):
    result = ray.get(increment.options(num_gpus=1).remote(1))
    assert result == 2

@pytest.mark.skipif(os.getenv("GPU_CI") != "true", reason="Skip this test if not running with a GPU in CI.")
def test_scheduling_with_scheduling_strategy_on_gpu(ray_cluster):
    result = ray.get(increment.options(scheduling_strategy=SINGLE_GPU_CONFIG.as_ray_scheduling_strategy()).remote(1))
    assert result == 2