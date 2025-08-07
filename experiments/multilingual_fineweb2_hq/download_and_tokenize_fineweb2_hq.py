"""
Tokenizes the Fineweb2-HQ dataset splits.

This module defines a function that returns tokenization steps for each dataset split available in the
Fineweb2 dataset.
"""

import os.path

from levanter.store.cache import CacheOptions

from experiments.llama import llama3_tokenizer
from experiments.multilingual_fineweb2_hq.constants import FINEWEB2_DATASETS
from marin.download.huggingface.download import DownloadConfig
from marin.download.huggingface.download_hf import download_hf
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

fineweb2_raw = ExecutorStep(
    name="raw/fineweb2_hq",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="epfml/FineWeb2-HQ",
        gcs_output_path=this_output_path(),
        revision="c0c06e94fd3a44ae9e802b2b0fc533817601eb5e",
        wait_for_completion=True,
    ),
).with_output_path("raw/fineweb2-hq")


def _get_fineweb2_split_paths(split):
    patterns = FINEWEB2_DATASETS[split]
    fineweb2_split_paths = [output_path_of(fineweb2_raw, pattern) for pattern in patterns]
    return fineweb2_split_paths


def tokenize_fineweb2hq_steps(*, base_path="tokenized/", tokenizer=llama3_tokenizer) -> dict[str, TokenizerStep]:
    steps = []
    for split in FINEWEB2_DATASETS.keys():
        fineweb2_split_output_path = os.path.join(base_path, "fineweb2_hq", split)
        fineweb2_split_paths = _get_fineweb2_split_paths(split)
        step = ExecutorStep(
            name=fineweb2_split_output_path,
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=fineweb2_split_paths,
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
                cache_options=CacheOptions(num_shard_groups=256),
            ),
            pip_dependency_groups=["sentencepiece"],
        )
        steps.append(step)
    return steps


if __name__ == "__main__":
    executor_main(steps=tokenize_fineweb2hq_steps(), description="Tokenize Fineweb2-HQ dataset")
