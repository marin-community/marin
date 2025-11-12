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

"""10M seed set runs for us-central2 using shared utils."""

from marin.execution.executor import executor_main

from experiments.memorize.utils import make_runner_10m

REGION = "central2"
runner = make_runner_10m(REGION)

# Representative epoch counts; adjust as needed
train_1epoch = runner(1)
train_3epoch = runner(3)
train_10epoch = runner(10)
train_20epoch = runner(20)
train_50epoch = runner(50)
train_60epoch = runner(60, timestamp="20251013_133633")
train_65epoch = runner(65, timestamp="20251013_133633")
train_75epoch = runner(75)
train_100epoch = runner(100)
train_200epoch = runner(200)
train_300epoch = runner(300, timestamp="20251013_133633")

train_375epoch = runner(375, timestamp="20251013_133633")
train_500epoch = runner(500, timestamp="20251013_133633")
train_750epoch = runner(750, timestamp="20251013_133633")
train_1000epoch = runner(1000, timestamp="20251013_133633")

if __name__ == "__main__":
    executor_main(
        steps=[
            # train_1epoch,
            #train_3epoch,
            #train_60epoch,
            #train_65epoch,
            # train_10epoch,
            # train_20epoch,
            # train_50epoch,
            # train_75epoch,
            # train_100epoch,
            # train_200epoch,
            #train_300epoch,
            train_375epoch,
            #train_500epoch,
            train_750epoch,
            train_1000epoch,
        ],
        description="150M on ~10M COMMA seed set (us-central2).",
    )
