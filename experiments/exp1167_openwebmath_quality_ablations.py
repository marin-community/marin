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

"""
This experiment evaluates the quality of openwebmath crawl data for model cooldown using `default_quality_ablation`
which fine-tunes an 8B model on a mixture of:
- 70% DCLM baseline data
- 15% openwebmath crawl data
- 15% Dolma/FLAN dataset

Reference Issue: https://github.com/stanford-crfm/marin/issues/1167
"""

from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.execution.executor import InputName, executor_main
from marin.resources import TpuPodConfig

# Tokenize the openwebmath dataset
openwebmath_crawl_tokenized = default_tokenize(
    "openwebmath-crawled",
    InputName.hardcoded(
        "gs://marin-us-central2/scratch/nfliu/text/open_web_math_100M_passing_minhash_against_open_web_math"
    ),
    tokenizer=llama3_tokenizer,
)

openwebmath_raw_tokenized = default_tokenize(
    "openwebmath-control",
    InputName.hardcoded(
        "gs://marin-us-central2/raw/open-web-math-fde8ef8/fde8ef8/huggingface.co/datasets/open-web-math/open-web-math/resolve/fde8ef8/data"
    ),
    tokenizer=llama3_tokenizer,
)

# Conduct the cooldown experiment over v4-128 TPU else the v5litepod-128
# TPU is used which is not available in us-central2
cooldown_config = QualityAblationConfig(
    resources=TpuPodConfig(tpu_type="v4-128"),
    permutation_type="linear",
)

# Conduct the cooldown experiment
openwebmath_crawl_ablation = default_quality_ablation(
    openwebmath_crawl_tokenized,
    cooldown_config,
)

openwebmath_raw_ablation = default_quality_ablation(
    openwebmath_raw_tokenized,
    cooldown_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            openwebmath_crawl_tokenized,
            openwebmath_crawl_ablation,
            openwebmath_raw_tokenized,
            openwebmath_raw_ablation,
        ],
    )
