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

Uses JAX-style tracing - steps call other steps naturally.
"""

from experiments.llama import llama3_tokenizer
from experiments.multilingual_fineweb2_hq.constants import FINEWEB2_DATASETS
from marin.download.huggingface.download_hf import DownloadConfig
from marin.download.huggingface.download_hf import download_hf as _download_hf
from marin.execution import deferred, executor_main, output, step, versioned
from marin.processing.tokenize import TokenizeConfig
from marin.processing.tokenize import tokenize as _tokenize

# Mark library functions as deferred
download_hf = deferred(_download_hf)
tokenize = deferred(_tokenize)


@step(name="raw/fineweb2_hq")
def fineweb2_raw():
    """Download the Fineweb2-HQ dataset from HuggingFace."""
    return download_hf(
        DownloadConfig(
            hf_dataset_id="epfml/FineWeb2-HQ",
            gcs_output_path=output(),
            revision="c0c06e94fd3a44ae9e802b2b0fc533817601eb5e",
            wait_for_completion=True,
        )
    )


@step(name="tokenized/fineweb2_hq/{split}")
def tokenize_fineweb2_split(split: str, tokenizer=llama3_tokenizer):
    """Tokenize a single Fineweb2-HQ split."""
    patterns = FINEWEB2_DATASETS[split]
    # Call fineweb2_raw() - returns StepRef, dependency auto-tracked
    raw = fineweb2_raw()
    train_paths = [raw / pattern for pattern in patterns]

    return tokenize(
        TokenizeConfig(
            train_paths=train_paths,
            validation_paths=versioned([]),
            cache_path=output(),
            tokenizer=versioned(tokenizer),
        )
    )


@step(name="fineweb2_hq/all")
def download_and_tokenize_all(tokenizer=llama3_tokenizer):
    """Entry point that downloads and tokenizes all FineWeb2-HQ datasets."""
    return [tokenize_fineweb2_split(split=split, tokenizer=tokenizer) for split in FINEWEB2_DATASETS.keys()]


if __name__ == "__main__":
    executor_main(steps=[download_and_tokenize_all()], description="Tokenize Fineweb2-HQ dataset")
