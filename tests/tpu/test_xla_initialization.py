# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple test that just checks whether scheduling a task on
a TPU works.
"""

import os

import pytest
import ray

from marin.generation.ray_utils import scheduling_strategy_fn


@ray.remote(resources={"TPU": 1})
def xla_check_device():
    import torch_xla.core.xla_model as xm

    return xm.get_xla_supported_devices()


@ray.remote(scheduling_strategy=scheduling_strategy_fn(1, "STRICT_PACK", "TPU-v6e", False))
def xla_check_device_with_scheduling_strategy():
    import torch_xla.core.xla_model as xm

    return xm.get_xla_supported_devices()


@pytest.mark.skipif(os.getenv("TPU_CI") != "true", reason="Skip this test if not running with a TPU in CI.")
def test_xla_initialization(ray_tpu_cluster):
    result = ray.get(xla_check_device.remote())
    assert result


@pytest.mark.skipif(os.getenv("TPU_CI") != "true", reason="Skip this test if not running with a TPU in CI.")
def test_xla_initialization_with_scheduling_strategy(ray_tpu_cluster):
    result = ray.get(xla_check_device_with_scheduling_strategy.remote())
    assert result


@ray.remote(resources={"TPU": 1})
class TPUActor:
    def __init__(self):
        self.tpu_id = ray.get_runtime_context().get_accelerator_ids()["TPU"][0]
        self.visible_chips = os.environ["TPU_VISIBLE_CHIPS"]

    def get_visible_chips(self):
        return self.visible_chips

    def ping(self):
        return self.tpu_id


@pytest.mark.skipif(os.getenv("TPU_CI") != "true", reason="Skip this test if not running with a TPU in CI.")
def test_not_conflicting_devices(ray_tpu_cluster):
    num_total_devices = 4

    xla_device_id_list = []
    actors = [None for _ in range(num_total_devices)]
    for i in range(num_total_devices):
        actors[i] = TPUActor.remote()
        result = ray.get(actors[i].ping.remote())
        assert result is not None

        if result not in xla_device_id_list:
            xla_device_id_list.append(result)

    assert len(xla_device_id_list) == num_total_devices
    assert set(xla_device_id_list) == set([str(i) for i in range(num_total_devices)])

    visible_chips_list = []
    for actor in actors:
        visible_actor_chip = ray.get(actor.get_visible_chips.remote())
        visible_chips_list.append(visible_actor_chip)

    assert len(visible_chips_list) == num_total_devices
    assert set(visible_chips_list) == set([str(i) for i in range(num_total_devices)])
