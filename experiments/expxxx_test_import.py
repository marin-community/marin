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

from marin.execution.executor import executor_main, ExecutorStep
import ray

# def direct(config):
#     return


@ray.remote(resources={"TPU-v4-8-head": 1})
def to_import(config):

    return ""


step = ExecutorStep(
    name="test_import",
    fn=to_import,
    config=None,
    pip_dependency_groups=["post_training", "vllm"],
)


if __name__ == "__main__":
    executor_main(
        steps=[
            step,
        ],
    )

    # ray.get(to_import.remote(None))
