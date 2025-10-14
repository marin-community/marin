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

"""100M seed set runs for us-central1 using shared utils.

Supports resuming specific runs by passing a fixed timestamp to the runner
for each step.
"""

import os

from marin.execution.executor import executor_main

from experiments.memorize.utils import make_runner_100m

REGION = "central1"
runner = make_runner_100m(REGION)



if __name__ == "__main__":
    # Build the desired steps here so we can pass the timestamp argument.
    steps = [
        #runner(200, timestamp="20251013_141440"),
        runner(300, timestamp="20251013_141440"),
        # runner(500, timestamp=RUNNER_TIMESTAMP),
        # runner(750, timestamp="20251013_141440"),
    ]

    executor_main(
        steps=steps,
        description="150M on ~100M COMMA seed set (us-central1).",
    )
