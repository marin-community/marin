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
This experiment evaluates the quality of fineweb-edu crawl data for model cooldown using `default_quality_ablation`
which fine-tunes an 8B model on a mixture of:
- 70% DCLM baseline data
- 15% fineweb-edu crawl data
- 15% Dolma/FLAN dataset

Reference Issue: https://github.com/stanford-crfm/marin/issues/1168
"""

from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.execution.executor import InputName, executor_main
from marin.resources import TpuPodConfig

# Tokenize the fineweb-edu dataset
fineweb_edu_tokenized = default_tokenize(
    "fineweb-edu-crawled",
    InputName.hardcoded(
        "gs://marin-us-central2/scratch/nfliu/text/fineweb_edu_unique_100M_passing_minhash_against_fineweb_edu"
    ),
    tokenizer=llama3_tokenizer,
)

fineweb_edu_raw_tokenized = default_tokenize(
    "fineweb-edu-control",
    InputName.hardcoded(
        "gs://marin-us-central2/raw/fineweb-edu-c2beb4/3c452cb/huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/3c452cb/data"
    ),
    tokenizer=llama3_tokenizer,
)

# Conduct the cooldown experiment over v4-128 TPU else the v5litepod-128
# TPU is used which is not available in us-central2
cooldown_config = QualityAblationConfig(
    resources=TpuPodConfig(tpu_type="v4-128"),
    permutation_type="linear",
)

# Conduct the cooldown experimentm
fineweb_edu_ablation = default_quality_ablation(
    fineweb_edu_tokenized,
    cooldown_config,
)

fineweb_edu_raw_ablation = default_quality_ablation(
    fineweb_edu_raw_tokenized,
    cooldown_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            fineweb_edu_tokenized,
            fineweb_edu_ablation,
            fineweb_edu_raw_tokenized,
            fineweb_edu_raw_ablation,
        ],
    )
