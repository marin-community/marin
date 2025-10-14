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

"""100M seed set runs for us-east5 using shared utils."""

from marin.execution.executor import executor_main

from experiments.memorize.utils import make_runner_100m

REGION = "east5"
runner = make_runner_100m(REGION)

# Define a broader set of epochs for convenience
train_1epoch = runner(1)
train_2epoch = runner(2)
train_5epoch = runner(5)
train_10epoch = runner(10)

# Active longer runs in this region
train_20epoch = runner(20)
train_50epoch = runner(50)
train_75epoch = runner(75)
train_100epoch = runner(100)
train_150epoch = runner(150)
train_200epoch = runner(200)
train_300epoch = runner(300)
train_500epoch = runner(500)
train_750epoch = runner(750)


if __name__ == "__main__":
    executor_main(
        #steps=[train_20epoch, train_50epoch, train_75epoch, train_100epoch, train_200epoch],
        #steps=[train_150epoch, train_300epoch, train_500epoch, train_750epoch],
        steps=[train_1epoch, train_2epoch, train_5epoch, train_10epoch, train_20epoch, train_50epoch, train_75epoch, train_100epoch],
        description="150M on ~100M COMMA seed set (us-east5).",
    )
