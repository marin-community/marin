# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DOLMINO dataset definitions and tokenization."""

import os.path

from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.datakit.download.dolmino import DOLMINO_DATASETS, download_dolmino_step
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

_dolmino_download = download_dolmino_step().as_executor_step()
_dolmino_base_dir = _dolmino_download.cd("bb54cab").cd("data")

# Backward compat — some consumers import this
downloads = {"dolmino": _dolmino_download.cd("bb54cab")}

# NB: we changed how hashes were computed for this corpus and we'd like to avoid recomputing them
DOLMINO_LLAMA3_OVERRIDES = {
    "dclm": "tokenized/dolmino/dclm-6c18eb",
    "flan": "tokenized/dolmino/flan-d71ec1",
    "math/codesearchnet-owmfilter": "tokenized/dolmino/math/codesearchnet-owmfilter-fd2640",
    "math/dolmino_math_synth": "tokenized/dolmino/math/dolmino_math_synth-11f876",
    "math/gsm8k": "tokenized/dolmino/math/gsm8k-902e8b",
    "math/mathcoder2-synthmath": "tokenized/dolmino/math/mathcoder2-synthmath-bc8dd2",
    "math/metamath-owmfilter": "tokenized/dolmino/math/metamath-owmfilter-fafa84",
    "math/tinyGSM-MIND": "tokenized/dolmino/math/tinyGSM-MIND-6c3016",
    "math/tulu_math": "tokenized/dolmino/math/tulu_math-414a4d",
    "pes2o": "tokenized/dolmino/pes2o-d22243",
    "stackexchange": "tokenized/dolmino/stackexchange-271a84",
    "wiki": "tokenized/dolmino/wiki-c31b74",
    "dolmino_dclm": "tokenized/dolmino/dclm-6c18eb",
}


def _get_dolmino_split_paths(split: str):
    """Helper to get file paths for a dolmino split."""
    patterns = DOLMINO_DATASETS[split]
    dolmino_split_input_base_path = _dolmino_base_dir / split
    return [dolmino_split_input_base_path / pattern for pattern in patterns]


def tokenize_dolmino(*, tokenizer: str | None = None) -> dict[str, TokenizerStep]:
    """Generate tokenization steps for all Dolmino dataset splits."""
    if tokenizer is None:
        from experiments.llama import llama3_tokenizer

        tokenizer = llama3_tokenizer

    dolmino_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for split in DOLMINO_DATASETS:
        dolmino_split_output_path = os.path.join("tokenized", "dolmino", split)
        dolmino_split_paths = _get_dolmino_split_paths(split)
        step = ExecutorStep(
            name=dolmino_split_output_path,
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=dolmino_split_paths,
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
            ),
        )

        # Check if we need to use override path for llama3
        from experiments.llama import llama3_tokenizer as _llama3_tokenizer

        if tokenizer == _llama3_tokenizer and split in DOLMINO_LLAMA3_OVERRIDES:
            step = step.with_output_path(DOLMINO_LLAMA3_OVERRIDES[split])
        dolmino_steps[os.path.join("dolmino", split)] = step

    return dolmino_steps


def tokenize_dolmino_subset(name: str, tokenizer: str | None = None) -> ExecutorStep[TokenizeConfig]:
    """Get a specific dolmino split tokenization step."""
    assert name in DOLMINO_DATASETS, f"Split {name} not found in DOLMINO_DATASETS"
    return tokenize_dolmino(tokenizer=tokenizer)[f"dolmino/{name}"]


# Special combined math split that includes all math/* datasets
_all_dolmino_math_files = [
    path for split in DOLMINO_DATASETS if "math" in split for path in _get_dolmino_split_paths(split)
]


def tokenize_dolmino_math(tokenizer: str | None = None):
    """Create the combined math dataset tokenization step."""
    if tokenizer is None:
        from experiments.llama import llama3_tokenizer

        tokenizer = llama3_tokenizer

    return ExecutorStep(
        name="tokenized/dolmino/all_math",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=_all_dolmino_math_files,
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
        ),
    ).with_output_path("tokenized/dolmino/all_math-9d507c")
