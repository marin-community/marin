# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dolmino dataset download definition and split metadata."""

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec

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


def download_dolmino_step() -> StepSpec:
    """Download the dolmino-mix-1124 dataset from HuggingFace."""
    return download_hf_step(
        "raw/dolmino-mix-1124",
        hf_dataset_id="allenai/dolmino-mix-1124",
        revision="bb54cab",
        override_output_path="raw/dolmino-mix-1124-157960",
    )
