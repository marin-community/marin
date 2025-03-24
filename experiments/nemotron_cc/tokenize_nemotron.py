"""
Tokenizes the Nemotron CC dataset splits.

This module defines a function that returns tokenization steps for each dataset split available in the
Nemotron CC dataset.
"""

import os.path

from levanter.store.cache import CacheOptions

from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets import nemotron_cc
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

nemotron_cc_path = output_path_of(nemotron_cc, "contrib/Nemotron/Nemotron-CC/data-jsonl/")

# The following dataset splits define file patterns for each split.
NEMOTRON_DATASETS = {
    "hq_actual": ["quality=high/kind=actual/**/*.jsonl.gz"],
    "hq_synth": ["quality=high/kind=synthetic/**/*.jsonl.gz"],
    "medium_high": ["quality=medium-high/**/*.jsonl.gz"],
    "medium": ["quality=medium/**/*.jsonl.gz"],
    "medium_low": ["quality=medium-low/**/*.jsonl.gz"],
    "low_actual": ["quality=low/kind=actual/**/*.jsonl.gz"],
    "low_synth": ["quality=low/kind=synthetic/**/*.jsonl.gz"],
}

# Weights for each split based on their size in TiB/GiB
# Converted GiB to TiB for consistency
NEMOTRON_WEIGHTS = {
    "nemotron_cc/hq_actual": 935.43 / 1024,  # 935.43 GiB
    "nemotron_cc/hq_synth": 2.72,  # 2.72 TiB
    "nemotron_cc/medium_high": 844.51 / 1024,  # 844.51 GiB
    "nemotron_cc/medium": 3.38,  # 3.38 TiB
    "nemotron_cc/medium_low": 1.54,  # 1.54 TiB
    "nemotron_cc/low_actual": 718.06 / 1024,  # 718.06 GiB
    "nemotron_cc/low_synth": 642.78 / 1024,  # 642.78 GiB
}


def tokenize_nemotron_steps(*, base_path="tokenized/", tokenizer=llama3_tokenizer) -> dict[str, TokenizerStep]:
    nemotron_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for split in NEMOTRON_DATASETS:
        nemotron_split_output_path = os.path.join(base_path, "nemotron_cc", split)
        nemotron_split_paths = _get_nemotron_split_paths(split)
        nemotron_steps[os.path.join("nemotron_cc", split)] = ExecutorStep(
            name=nemotron_split_output_path,
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=nemotron_split_paths,
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
                cache_options=CacheOptions(num_shard_groups=256),
            ),
            pip_dependency_groups=["sentencepiece"],
        )

    assert nemotron_steps.keys() == NEMOTRON_WEIGHTS.keys()
    return nemotron_steps


def _get_nemotron_split_paths(split):
    patterns = NEMOTRON_DATASETS[split]
    nemotron_split_paths = [nemotron_cc_path / pattern for pattern in patterns]
    return nemotron_split_paths


def get_nemotron_step(split: str) -> ExecutorStep[TokenizeConfig]:
    assert (
        split in NEMOTRON_DATASETS
    ), f"Split {split} not found in {NEMOTRON_DATASETS}, \
        Check marin.experiments.nemotron_cc.tokenize_nemotron.NEMOTRON_DATASETS for which splits are supported."
    return tokenize_nemotron_steps()[f"nemotron_cc/{split}"]


if __name__ == "__main__":
    executor_main(steps=list(tokenize_nemotron_steps().values()), description="Tokenize Nemotron dataset")
