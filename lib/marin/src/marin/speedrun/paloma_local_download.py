# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
The Paloma eval sets, downloaded and tokenized

https://huggingface.co/datasets/allenai/paloma
"""

from experiments.paloma import paloma_tokenized
from marin.download import HfDownloadConfig
from marin.download.huggingface.download_hf import download_hf
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"

paloma_speedrun = ExecutorStep(
    name="raw/paloma-speedrun",
    fn=download_hf,
    config=HfDownloadConfig(
        hf_dataset_id=versioned("allenai/paloma"),
        revision=versioned("65cd6fc"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        append_sha_to_path=True,
    ),
).cd("65cd6fc")


def speedrun_paloma_tokenized(tokenizer: str = llama3_tokenizer):
    return paloma_tokenized(base_path="raw/paloma-speedrun", tokenizer=tokenizer, paloma_raw=paloma_speedrun)


if __name__ == "__main__":
    executor_main(steps=[paloma_speedrun, *speedrun_paloma_tokenized])
