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

import dataclasses

from levanter.data.text import TextLmDatasetFormat

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets import (
    dclm_baseline_tokenized_llama3,
    dclm_baseline_wrong,
    proofpile_2_tokenized_llama3,
    starcoderdata_tokenized_llama3,
)
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

DCLM_MIXTURE_WEIGHTS = {
    # token counts are for neox tokenizer
    "dclm_baseline": 3.8,  # 3.8 trillion tokens https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
    "starcoderdata": 0.25,  # 250 billion tokens https://huggingface.co/datasets/bigcode/starcoderdata
    "proofpile_2": 0.055,  # 55 billion tokens https://huggingface.co/datasets/EleutherAI/proof-pile-2
}


DCLM_BASELINE_ONLY_MIXTURE = {
    "dclm_baseline": 3.8,  # 3.8 trillion tokens https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
    "starcoderdata": 0,  # 250 billion tokens https://huggingface.co/datasets/bigcode/starcoderdata
    "proofpile_2": 0,  # 55 billion tokens https://huggingface.co/datasets/EleutherAI/proof-pile-2
}


dclm_components_llama3 = {
    "dclm_baseline": dclm_baseline_tokenized_llama3,
    "starcoderdata": starcoderdata_tokenized_llama3,
    "proofpile_2": proofpile_2_tokenized_llama3,
}
dclm_mixture_config_llama3_old = lm_mixture_data_config(
    components=dclm_components_llama3,
    weights=DCLM_MIXTURE_WEIGHTS,
    permutation_type="linear",
)

dclm_mixture_config_llama3 = lm_mixture_data_config(
    components=dclm_components_llama3, weights=DCLM_MIXTURE_WEIGHTS, permutation_type="feistel"
)


## NOTE: on 20250211, we discovered that the DCLM baseline data in us-central2 was corrupted/partial.
# These are preserved for reproducibility, but future runs should use the correct data.
# YOU SHOULD NOT USE THESE TOKENIZED DATASETS FOR TRAINING
dclm_components_llama3_wrong = {
    "dclm_baseline": dataclasses.replace(
        default_tokenize(
            name="dclm_baseline",
            dataset=dclm_baseline_wrong,
            tokenizer=llama3_tokenizer,
        ),
        override_output_path="gs://marin-us-central2/tokenized/dclm_baseline-0206f1_WRONG_20250211/",
    ),
    "starcoderdata": default_tokenize(
        name="starcoderdata",
        dataset=starcoderdata,
        tokenizer=llama3_tokenizer,
        format=TextLmDatasetFormat(text_key="content"),
    ),
    "proofpile_2": default_tokenize(
        name="proofpile_2",
        dataset=proofpile_2,
        tokenizer=llama3_tokenizer,
    ),
}

dclm_mixture_config_llama3_wrong = lm_mixture_data_config(
    components=dclm_components_llama3_wrong,
    weights=DCLM_MIXTURE_WEIGHTS,
    permutation_type="linear",
)

if __name__ == "__main__":
    executor_main(steps=list(dclm_components_llama3.values()))
