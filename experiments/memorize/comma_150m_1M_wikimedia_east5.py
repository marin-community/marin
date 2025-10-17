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

"""1M (Wikimedia-only) seed set runs for us-east5 using shared utils."""

from marin.execution.executor import executor_main

from experiments.memorize.utils import make_runner_1m_wikimedia

REGION = "east5"
runner = make_runner_1m_wikimedia(REGION)

# Baseline epoch counts for 1M seed set
train_1epoch = runner(1)
train_10epoch = runner(10)
train_20epoch = runner(20)
train_50epoch = runner(50)
train_75epoch = runner(75)
train_100epoch = runner(100)
train_200epoch = runner(200)


if __name__ == "__main__":
    executor_main(
        steps=[
            train_1epoch,
            train_10epoch,
            train_20epoch,
            train_50epoch,
            train_75epoch,
            train_100epoch,
            train_200epoch,
        ],
        description="150M on ~1M Wikimedia-only seed set (us-east5).",
    )
