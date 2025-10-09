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

"""Prewarm Common Pile tokenized caches in us-east5.

This runs the Common Pile tokenization steps only, writing outputs under the
`--prefix` you pass (e.g., `gs://marin-us-east5`). Use it to stage data in-region
before training, avoiding cross-region reads.

Launch with (Ray Jobs on us-east5 cluster):
    uv run python src/marin/run/ray_run.py --cluster infra/marin-us-east5.yaml -- \
        python experiments/common_pile/prewarm_us_east5.py --prefix gs://marin-us-east5 --force_run_failed True

Tip: To prewarm a subset, use `--run-only` with a regex, e.g.:
    --run-only '^tokenized/common_pile/wikimedia$'
"""

from experiments.common_pile.tokenize_common_pile import common_pile_tokenized
from experiments.llama import llama3_tokenizer
from marin.execution.executor import executor_main

if __name__ == "__main__":
    steps = list(common_pile_tokenized(tokenizer=llama3_tokenizer).values())
    executor_main(steps=steps, description="Prewarm Common Pile tokenized datasets in us-east5")
