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
Script to create tokenized subcaches for speedrun.

This script is used to create subcaches of the fineweb-edu dataset for use in Marin Speedrun.

Running this experiment will create two subcaches:
1. A 10B token subcache, which is a subset of the original fineweb-edu dataset consisting of approximately 10B tokens.
2. A 10M token subcache, which is a smaller subset of the original fineweb-edu dataset. (Mostly for testing purposes)

You probably don't need to run this script unless you're adding a new dataset or changing the tokenization process.
"""

from experiments.exp524_tokenizers import fineweb_edu_llama3_tokenized
from experiments.speedrun.prebuilt_caches import fineweb_edu_10B_repo_id, fineweb_edu_10M_repo_id
from experiments.steps import upload_dir_to_hf
from marin.execution import executor_main
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.tokenize.slice_cache import slice_cache

base_cache = fineweb_edu_llama3_tokenized

fineweb_edu_subcache_10B_created = slice_cache(
    output_path="tokenized/subcache/fineweb-edu-10B",
    input_config=step_to_lm_mixture_component(fineweb_edu_llama3_tokenized, include_raw_paths=True),
    num_tokens=10_000_000_000,
)
uploaded_cert_10B = upload_dir_to_hf(fineweb_edu_subcache_10B_created, repo_id=fineweb_edu_10B_repo_id)

fineweb_edu_subcache_10M_created = slice_cache(
    output_path="tokenized/subcache/fineweb-edu-10M",
    input_config=step_to_lm_mixture_component(fineweb_edu_llama3_tokenized, include_raw_paths=True),
    num_tokens=10_000_000,
)

uploaded_cert_10M = upload_dir_to_hf(fineweb_edu_subcache_10M_created, repo_id=fineweb_edu_10M_repo_id)


if __name__ == "__main__":
    executor_main(
        steps=[fineweb_edu_subcache_10B_created, uploaded_cert_10B, fineweb_edu_subcache_10M_created, uploaded_cert_10M],
        description="Create subcaches of the fineweb-edu dataset.",
    )
