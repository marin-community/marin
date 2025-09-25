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

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


# Used in normal ray remote args scheduling strategies
def scheduling_strategy_fn(
    tensor_parallel_size: int, strategy: str, tpu_type: str, include_head_in_scheduling_strategy: bool
):

    # One bundle per tensor parallel worker
    tpus_and_cpus = [{"TPU": 1, "CPU": 1}] * tensor_parallel_size
    tpu_head = [{f"{tpu_type}-head": 1}]

    pg = ray.util.placement_group(
        tpus_and_cpus + tpu_head,
        strategy=strategy,  # STRICT_PACK means same node, PACK can span multiple nodes
    )
    return PlacementGroupSchedulingStrategy(pg, placement_group_capture_child_tasks=True)


# Used in Ray Data pipelines
def get_ray_remote_args_scheduling_strategy_fn(tensor_parallel_size: int, strategy: str):
    def scheduling_strategy_dict_fn():
        return dict(scheduling_strategy=scheduling_strategy_fn(tensor_parallel_size, strategy))

    return scheduling_strategy_dict_fn
