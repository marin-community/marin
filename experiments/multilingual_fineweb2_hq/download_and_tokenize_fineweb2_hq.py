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

"""
Tokenizes the Fineweb2-HQ dataset splits.

This module defines a function that returns tokenization steps for each dataset split available in the
Fineweb2 dataset.
"""

import os.path


from experiments.llama import llama3_tokenizer
from experiments.multilingual_fineweb2_hq.constants import FINEWEB2_DATASETS
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution import step, StepContext, executor_main, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

@step(name="raw/fineweb2_hq", fn=download_hf)
def fineweb2_raw_creator(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="epfml/FineWeb2-HQ",
        gcs_output_path=ctx.output,
        revision="c0c06e94fd3a44ae9e802b2b0fc533817601eb5e",
        wait_for_completion=True,
    )


fineweb2_raw = fineweb2_raw_creator().with_output_path("raw/fineweb2-hq")


def _get_fineweb2_split_paths(split):
    patterns = FINEWEB2_DATASETS[split]
    fineweb2_split_paths = [fineweb2_raw / pattern for pattern in patterns]
    return fineweb2_split_paths


def _create_tokenize_step(split, base_path, tokenizer):
    """Helper function to create a tokenize step for a single split."""
    fineweb2_split_output_path = os.path.join(base_path, "fineweb2_hq", split)
    fineweb2_split_paths = _get_fineweb2_split_paths(split)

    @step(name=fineweb2_split_output_path, fn=tokenize)
    def tokenize_step_creator(ctx: StepContext):
        return TokenizeConfig(
            train_paths=fineweb2_split_paths,
            validation_paths=versioned([]),
            cache_path=ctx.output,
            tokenizer=versioned(tokenizer),
        )

    return tokenize_step_creator()


def tokenize_fineweb2hq_steps(*, base_path="tokenized/", tokenizer=llama3_tokenizer) -> dict[str, TokenizerStep]:
    """Return a mapping from dataset key to tokenization step for Fineweb2-HQ.

    Keys follow the pattern "fineweb2_hq/<split>", aligning with mixture naming conventions
    in other datasets (e.g., "dolma/...", "nemotron_cc/...").
    """
    steps: dict[str, TokenizerStep] = {}
    for split in FINEWEB2_DATASETS.keys():
        step = _create_tokenize_step(split, base_path, tokenizer)
        steps[f"fineweb2_hq/{split}"] = step
    return steps


if __name__ == "__main__":
    executor_main(steps=list(tokenize_fineweb2hq_steps().values()), description="Tokenize Fineweb2-HQ dataset")
