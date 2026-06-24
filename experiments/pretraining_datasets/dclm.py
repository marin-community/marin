# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from levanter.data.text import TextLmDatasetFormat
from marin.execution import executor_context
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.simple import downloads, tokenized
from experiments.tokenization import default_tokenize

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


def dclm_components_llama3() -> dict:
    tok = tokenized()
    return {
        "dclm_baseline": tok["dclm_baseline"],
        "starcoderdata": tok["starcoderdata"],
        "proofpile_2": tok["proofpile_2"],
    }


def dclm_mixture_config_llama3():
    return lm_mixture_data_config(components=dclm_components_llama3(), weights=DCLM_MIXTURE_WEIGHTS)


## NOTE: on 20250211, we discovered that the DCLM baseline data in us-central2 was corrupted/partial.
# These are preserved for reproducibility, but future runs should use the correct data.
# YOU SHOULD NOT USE THESE TOKENIZED DATASETS FOR TRAINING
def dclm_components_llama3_wrong() -> dict:
    raw = downloads()
    return {
        "dclm_baseline": dataclasses.replace(
            default_tokenize(
                name="dclm_baseline",
                dataset=raw["dclm_baseline_wrong"],
                tokenizer=llama3_tokenizer,
            ),
            override_output_path="gs://marin-us-central2/tokenized/dclm_baseline-0206f1_WRONG_20250211/",
        ),
        "starcoderdata": default_tokenize(
            name="starcoderdata",
            dataset=raw["starcoderdata"],
            tokenizer=llama3_tokenizer,
            format=TextLmDatasetFormat(text_key="content"),
        ),
        "proofpile_2": default_tokenize(
            name="proofpile_2",
            dataset=raw["proofpile_2"],
            tokenizer=llama3_tokenizer,
        ),
    }


def dclm_mixture_config_llama3_wrong():
    return lm_mixture_data_config(
        components=dclm_components_llama3_wrong(),
        weights=DCLM_MIXTURE_WEIGHTS,
    )


if __name__ == "__main__":
    with executor_context():
        executor_main(steps=list(dclm_components_llama3().values()))
