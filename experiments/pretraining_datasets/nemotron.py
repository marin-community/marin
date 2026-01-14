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

"""NEMOTRON CC dataset definitions and tokenization."""

import os.path

from marin.download.nemotron_cc.download_nemotron_cc import NemotronIngressConfig, download_nemotron_cc
from marin.execution.executor import ExecutorStep, StepRef, versioned
from marin.execution import step, StepContext
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

# Raw dataset download step
@step(name="raw/nemotro-cc", fn=download_nemotron_cc)
def _nemotron_cc_download(ctx: StepContext):
    return NemotronIngressConfig(
        output_path=ctx.output,
    )


downloads = {"nemotron_cc": _nemotron_cc_download()}

_nemotron_cc_path = downloads["nemotron_cc"] / "contrib/Nemotron/Nemotron-CC/data-jsonl/"

NEMOTRON_DATASETS = {
    "hq_actual": ["quality=high/kind=actual/**/*.jsonl.gz"],
    "hq_synth": ["quality=high/kind=synthetic/**/*.jsonl.gz"],
    "medium_high": ["quality=medium-high/**/*.jsonl.gz"],
    "medium": ["quality=medium/**/*.jsonl.gz"],
    "medium_low": ["quality=medium-low/**/*.jsonl.gz"],
    "low_actual": ["quality=low/kind=actual/**/*.jsonl.gz"],
    "low_synth": ["quality=low/kind=synthetic/**/*.jsonl.gz"],
}

# Weights for each split based on their size in TiB
NEMOTRON_WEIGHTS = {
    "nemotron_cc/hq_actual": 0.91351,  # TiB
    "nemotron_cc/hq_synth": 2.72,  # TiB
    "nemotron_cc/medium_high": 0.82471,  # TiB
    "nemotron_cc/medium": 3.38,  # TiB
    "nemotron_cc/medium_low": 1.54,  # TiB
    "nemotron_cc/low_actual": 0.70123,  # TiB
    "nemotron_cc/low_synth": 0.62771,  # TiB
}

# NB: we changed how hashes were computed for this corpus and we'd like to avoid recomputing them
NEMOTRON_LLAMA3_OVERRIDES = {
    "hq_actual": "tokenized/nemotron_cc/hq_actual-5af4cc",
    "hq_synth": "tokenized/nemotron_cc/hq_synth-3525e2",
    "low_actual": "tokenized/nemotron_cc/low_actual-cb3f2c",
    "low_synth": "tokenized/nemotron_cc/low_synth-3c57b3",
    "medium": "tokenized/nemotron_cc/medium-d86506",
    "medium_high": "tokenized/nemotron_cc/medium_high-d21701",
    "medium_low": "tokenized/nemotron_cc/medium_low-0fdb07",
}


def _get_nemotron_split_paths(split: str):
    """Helper to get file paths for a nemotron split."""
    patterns = NEMOTRON_DATASETS[split]
    return [_nemotron_cc_path / pattern for pattern in patterns]


def tokenize_nemotron(*, tokenizer: str | None = None) -> dict[str, TokenizerStep]:
    """Generate tokenization steps for all Nemotron CC dataset splits."""
    if tokenizer is None:
        from experiments.llama import llama3_tokenizer

        tokenizer = llama3_tokenizer

    nemotron_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for split in NEMOTRON_DATASETS:
        nemotron_split_output_path = os.path.join("tokenized", "nemotron_cc", split)
        nemotron_split_paths = _get_nemotron_split_paths(split)

        @step(name=nemotron_split_output_path, fn=tokenize)
        def tokenize_nemotron_split(ctx: StepContext, paths=nemotron_split_paths, tok=tokenizer):
            return TokenizeConfig(
                train_paths=paths,
                validation_paths=versioned([]),
                cache_path=ctx.output,
                tokenizer=versioned(tok),
            )

        result = tokenize_nemotron_split()

        # Check if we need to use override path for llama3
        from experiments.llama import llama3_tokenizer as _llama3_tokenizer

        if tokenizer == _llama3_tokenizer and split in NEMOTRON_LLAMA3_OVERRIDES:
            result = result.with_output_path(NEMOTRON_LLAMA3_OVERRIDES[split])

        nemotron_steps[os.path.join("nemotron_cc", split)] = result

    assert nemotron_steps.keys() == NEMOTRON_WEIGHTS.keys()
    return nemotron_steps


def tokenize_nemotron_subset(name: str, tokenizer: str | None = None) -> ExecutorStep[TokenizeConfig]:
    """Get a specific nemotron split tokenization step."""
    assert name in NEMOTRON_DATASETS, f"Split {name} not found in NEMOTRON_DATASETS"
    return tokenize_nemotron(tokenizer=tokenizer)[f"nemotron_cc/{name}"]
