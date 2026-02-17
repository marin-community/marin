# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DOLMINO dataset definitions and tokenization."""

import os
import os.path

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.step_model import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

# Raw dataset download step
_dolmino_base = StepSpec(
    name="raw/dolmino-mix-1124",
    hash_attrs={"hf_dataset_id": "allenai/dolmino-mix-1124", "revision": "bb54cab"},
    override_output_path="raw/dolmino-mix-1124-157960",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="allenai/dolmino-mix-1124",
            revision="bb54cab",
            gcs_output_path=output_path,
            wait_for_completion=True,
        )
    ),
)

downloads = {
    "dolmino": os.path.join(_dolmino_base.output_path, "bb54cab"),
}

_dolmino_data_dir = os.path.join(downloads["dolmino"], "data")

# The following dataset splits define file patterns for each split.
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


def _get_dolmino_split_paths(split: str) -> list[str]:
    """Helper to get file paths for a dolmino split."""
    patterns = DOLMINO_DATASETS[split]
    dolmino_split_input_base_path = os.path.join(_dolmino_data_dir, split)
    return [os.path.join(dolmino_split_input_base_path, pattern) for pattern in patterns]


def tokenize_dolmino(*, tokenizer: str | None = None) -> dict[str, TokenizerStep]:
    """Generate tokenization steps for all Dolmino dataset splits."""
    if tokenizer is None:
        from experiments.llama import llama3_tokenizer

        tokenizer = llama3_tokenizer

    dolmino_steps: dict[str, StepSpec] = {}
    for split in DOLMINO_DATASETS:
        dolmino_split_output_path = os.path.join("tokenized", "dolmino", split)
        dolmino_split_paths = _get_dolmino_split_paths(split)

        override = None
        from experiments.llama import llama3_tokenizer as _llama3_tokenizer

        if tokenizer == _llama3_tokenizer and split in DOLMINO_LLAMA3_OVERRIDES:
            override = DOLMINO_LLAMA3_OVERRIDES[split]

        # Capture loop variables for the closure
        _paths = dolmino_split_paths
        _tokenizer = tokenizer

        step = StepSpec(
            name=dolmino_split_output_path,
            hash_attrs={"tokenizer": tokenizer, "validation_paths": []},
            override_output_path=override,
            fn=lambda output_path, _p=_paths, _tk=_tokenizer: tokenize(
                TokenizeConfig(
                    train_paths=_p,
                    validation_paths=[],
                    cache_path=output_path,
                    tokenizer=_tk,
                )
            ),
        )

        dolmino_steps[os.path.join("dolmino", split)] = step

    return dolmino_steps


def tokenize_dolmino_subset(name: str, tokenizer: str | None = None) -> StepSpec:
    """Get a specific dolmino split tokenization step."""
    assert name in DOLMINO_DATASETS, f"Split {name} not found in DOLMINO_DATASETS"
    return tokenize_dolmino(tokenizer=tokenizer)[f"dolmino/{name}"]


# Special combined math split that includes all math/* datasets
_all_dolmino_math_files = [
    path for split in DOLMINO_DATASETS if "math" in split for path in _get_dolmino_split_paths(split)
]


def tokenize_dolmino_math(tokenizer: str | None = None) -> StepSpec:
    """Create the combined math dataset tokenization step."""
    if tokenizer is None:
        from experiments.llama import llama3_tokenizer

        tokenizer = llama3_tokenizer

    _tokenizer = tokenizer
    _paths = _all_dolmino_math_files

    return StepSpec(
        name="tokenized/dolmino/all_math",
        hash_attrs={"tokenizer": tokenizer, "validation_paths": []},
        override_output_path="tokenized/dolmino/all_math-9d507c",
        fn=lambda output_path: tokenize(
            TokenizeConfig(
                train_paths=_paths,
                validation_paths=[],
                cache_path=output_path,
                tokenizer=_tokenizer,
            )
        ),
    )
