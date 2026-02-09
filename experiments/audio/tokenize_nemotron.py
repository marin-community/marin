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

import os.path

from experiments.pretraining_datasets.nemotron import downloads as nemotron_downloads
from experiments.llama import llama3_tokenizer
from marin.execution import versioned
from marin.execution.executor import ExecutorStep, this_output_path, output_path_of
from marin.processing.tokenize.data_configs import TokenizerStep
from marin.processing.tokenize import TokenizeConfig, tokenize

YODAS2_TOKENIZER = "potsawee/marin-mimi-bpe-8cb-16k-tokenizer"
YODAS2_QWEN_TOKENIZER = "potsawee/qwen3-mimi-bpe-8cb-16k-tokenizer"

nemotron_cc = nemotron_downloads["nemotron_cc"]
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

# NB: we changed how hashes were computed for this corpus and we'd like to avoid recomputing them
NEMOTRON_LLAMA3_OVERIDES = {
    "hq_actual": "tokenized/nemotron_cc/hq_actual-5af4cc",
    "hq_synth": "tokenized/nemotron_cc/hq_synth-3525e2",
    "low_actual": "tokenized/nemotron_cc/low_actual-cb3f2c",
    "low_synth": "tokenized/nemotron_cc/low_synth-3c57b3",
    "medium": "tokenized/nemotron_cc/medium-d86506",
    "medium_high": "tokenized/nemotron_cc/medium_high-d21701",
    "medium_low": "tokenized/nemotron_cc/medium_low-0fdb07",
}


def _get_nemotron_split_paths(split):
    patterns = NEMOTRON_DATASETS[split]
    nemotron_split_paths = [nemotron_cc_path / pattern for pattern in patterns]
    return nemotron_split_paths


def tokenize_nemotron_hq_actual_step(*, base_path="tokenized/", tokenizer=YODAS2_TOKENIZER) -> TokenizerStep:
    split = "hq_actual"
    nemotron_split_output_path = os.path.join(base_path, "nemotron_cc", split)
    nemotron_split_paths = _get_nemotron_split_paths(split)
    step = ExecutorStep(
        name=nemotron_split_output_path,
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=nemotron_split_paths,
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
        ),
        pip_dependency_groups=["tokenize_train"],
    )
    if tokenizer in [llama3_tokenizer, YODAS2_TOKENIZER] and split in NEMOTRON_LLAMA3_OVERIDES:
        step = step.with_output_path(NEMOTRON_LLAMA3_OVERIDES[split])
    return step
