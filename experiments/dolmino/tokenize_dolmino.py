"""
Tokenizes the Dolmino dataset splits.

This module defines a function that returns tokenization steps for each dataset split available in the dolmino dataset.
"""

import os.path

from levanter.store.cache import CacheOptions

from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

BASE_DIR_DOLMINO = "gs://marin-us-central2/raw/dolmino-mix-1124-157960/bb54cab/data"

# The following dataset splits define file patterns for each split.
# The glob pattern to capture all files with the extension under the folder.
DOLMINO_DATASETS = {
    "dclm": ["**/*.json.zst"],
    "flan": ["**/*.json.gz"],
    "math/codesearchnet-owmfilter": ["**/*.jsonl.gz"],
    "math/dolmino_math_synth": ["**/*.jsonl"],
    "math/gsm8k": ["**/*.jsonl.zst"],
    "math/mathcoder2-synthmath": ["**/*.jsonl"],
    "math/metamath-owmfilter": ["**/*.jsonl.gz"],
    "math/tinyGSM-MIND": ["**/*.jsonl.gz"],
    "math/tulu_math": ["**/*.jsonl"],
    "pes2o": ["**/*.json.gz"],
    "stackexchange": ["**/*.json.gz"],
    "wiki": ["**/*.json.gz"],
}


def tokenize_dolmino_steps(*, base_path="tokenized/", tokenizer=llama3_tokenizer) -> dict[str, TokenizerStep]:
    dolmino_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for split in DOLMINO_DATASETS:
        dolmino_split_output_path = os.path.join(base_path, "dolmino", split)
        dolmino_split_paths = _get_dolmino_split_paths(split)
        dolmino_steps[os.path.join("dolmino", split)] = ExecutorStep(
            name=dolmino_split_output_path,
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=versioned(dolmino_split_paths),
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
            ),
            pip_dependency_groups=["sentencepiece"],
        )
    return dolmino_steps


def _get_dolmino_split_paths(split):
    patterns = DOLMINO_DATASETS[split]
    dolmino_split_input_base_path = os.path.join(BASE_DIR_DOLMINO, split)
    dolmino_split_paths = [f"{dolmino_split_input_base_path}/{pattern}" for pattern in patterns]
    return dolmino_split_paths


all_dolmino_math_files = [
    path for split in DOLMINO_DATASETS if "math" in split for path in _get_dolmino_split_paths(split)
]

dolmino_math_tokenized_llama3 = ExecutorStep(
    name="tokenized/dolmino/all_math",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=versioned(all_dolmino_math_files),
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
        cache_options=CacheOptions(num_shard_groups=32),
    ),
    pip_dependency_groups=["sentencepiece"],
)


def get_dolmino_step(split: str) -> ExecutorStep[TokenizeConfig]:
    assert (
        split in DOLMINO_DATASETS
    ), f"Split {split} not found in {DOLMINO_DATASETS}, \
        Check marin.experiments.dolmino.tokenize_dolmino.DOLMINO_DATASETS for which splits are supported."
    return tokenize_dolmino_steps()[f"dolmino/{split}"]


if __name__ == "__main__":
    executor_main(steps=list(tokenize_dolmino_steps().values()))
