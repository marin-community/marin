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

"""100M seed set runs for us-central2 using shared utils."""

from marin.execution.executor import executor_main

from experiments.memorize.utils import make_runner_100m

REGION = "central2"
runner = make_runner_100m(REGION)

train_1epoch = runner(1)
train_2epoch = runner(2)
train_5epoch = runner(5)
train_10epoch = runner(10)


if __name__ == "__main__":
    executor_main(
        steps=[train_1epoch, train_2epoch, train_5epoch, train_10epoch],
        description="150M on ~100M COMMA seed set (us-central2).",
    )
