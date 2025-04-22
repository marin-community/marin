"""Simple test that just checks whether scheduling a task on
a TPU works.
"""

import os

import pytest
import ray


@ray.remote(resources={"TPU": 1})
def increment(x):
    return x + 1


@pytest.mark.skipif(os.getenv("TPU_CI") != "true", reason="Skip this test if not running with a TPU in CI.")
def test_scheduling_on_tpu(ray_tpu_cluster):
    result = ray.get(increment.remote(1))
    assert result == 2
