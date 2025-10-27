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

# Copyright 2025 The Marin Authors

"""10M seed set runs for us-east5 using shared utils."""

from marin.execution.executor import executor_main

from experiments.memorize.utils import make_runner_10m

REGION = "east5"
runner = make_runner_10m(REGION)

# Active longer runs for this region

train_500epoch = runner(500)
train_750epoch = runner(750)
train_1000epoch = runner(1000)
train_1500epoch = runner(1500)
train_3000epoch = runner(3000)

train_60epoch = runner(60)
train_65epoch = runner(65)

if __name__ == "__main__":
    executor_main(
        steps=[
            # train_500epoch,
            # train_750epoch,
            # train_1000epoch,
            # train_1500epoch,
            # train_3000epoch,
            train_60epoch,
            train_65epoch,
        ],
        description="150M on ~10M COMMA seed set (us-east5).",
    )
