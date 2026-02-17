# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
The Paloma eval sets, downloaded and tokenized

https://huggingface.co/datasets/allenai/paloma
"""

from experiments.paloma import paloma_tokenized
from marin.download import HfDownloadConfig
from marin.download.huggingface.download_hf import download_hf
from marin.execution.step_model import StepSpec
from marin.execution.step_runner import StepRunner

llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"

paloma_speedrun = StepSpec(
    name="raw/paloma-speedrun",
    hash_attrs={"hf_dataset_id": "allenai/paloma", "revision": "65cd6fc"},
    fn=lambda output_path: download_hf(
        HfDownloadConfig(
            hf_dataset_id="allenai/paloma",
            revision="65cd6fc",
            gcs_output_path=output_path,
            wait_for_completion=True,
            append_sha_to_path=True,
        )
    ),
)


def speedrun_paloma_tokenized(tokenizer: str = llama3_tokenizer):
    return paloma_tokenized(tokenizer=tokenizer)


if __name__ == "__main__":
    StepRunner().run([paloma_speedrun, *speedrun_paloma_tokenized().values()])
