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

"""DOLMINO dataset definitions and tokenization."""

import os.path

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, StepRef, versioned
from marin.execution import step, StepContext
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

# Raw dataset download step
@step(name="raw/dolmino-mix-1124", fn=download_hf, override_output_path="raw/dolmino-mix-1124-157960")
def dolmino_download(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="allenai/dolmino-mix-1124",
        revision="bb54cab",
        gcs_output_path=ctx.output,
        wait_for_completion=True,
    )


def _get_dolmino_base_dir():
    """Get the base directory for dolmino dataset."""
    return dolmino_download().cd("bb54cab").cd("data")

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


def _get_dolmino_split_paths(split: str):
    """Helper to get file paths for a dolmino split."""
    patterns = DOLMINO_DATASETS[split]
    dolmino_split_input_base_path = _get_dolmino_base_dir() / split
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

        @step(name=dolmino_split_output_path, fn=tokenize)
        def tokenize_dolmino_split(ctx: StepContext, paths=dolmino_split_paths, tok=tokenizer):
            return TokenizeConfig(
                train_paths=paths,
                validation_paths=versioned([]),
                cache_path=ctx.output,
                tokenizer=versioned(tok),
            )

        result = tokenize_dolmino_split()

        # Check if we need to use override path for llama3
        from experiments.llama import llama3_tokenizer as _llama3_tokenizer

        if tokenizer == _llama3_tokenizer and split in DOLMINO_LLAMA3_OVERRIDES:
            result = result.with_output_path(DOLMINO_LLAMA3_OVERRIDES[split])
        dolmino_steps[os.path.join("dolmino", split)] = result

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

    @step(name="tokenized/dolmino/all_math", fn=tokenize)
    def tokenize_dolmino_all_math(ctx: StepContext):
        return TokenizeConfig(
            train_paths=_all_dolmino_math_files,
            validation_paths=versioned([]),
            cache_path=ctx.output,
            tokenizer=versioned(tokenizer),
        )

    return tokenize_dolmino_all_math().with_output_path("tokenized/dolmino/all_math-9d507c")
