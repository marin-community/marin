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

"""Tokenize all datasets with Qwen3x (Qwen3 tokenizer + <|begin_of_text|> and <|end_of_text|> tokens)"""

import os
from collections.abc import Mapping

from experiments.defaults import default_tokenize
from marin.download.huggingface.download_hf import DownloadConfig, download_hf as storage_transfer_download
from marin.execution import executor_main, versioned
from marin.execution.executor import ExecutorStep, this_output_path, output_path_of
from experiments.pretraining_datasets.nemotron import downloads as nemotron_downloads
from marin.processing.tokenize import tokenize, TokenizeConfig

QWEN3X_TOKENIZER = "potsawee/qwen3x-mimi-bpe-8cb-16k-tokenizer"


# Emilia
_EMILIA_REVISION_STR = "e41392e668b273145344fda2f737cd75a6ebb558"
EMILIA_REVISION = versioned(_EMILIA_REVISION_STR)
_EMILIA_SPLIT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Emilia/EN", "Emilia/EN/*.parquet"),
    ("Emilia-YODAS/EN", "Emilia-YODAS/EN/*.parquet"),
)
_EMILIA_DOWNLOAD_NAME = "raw/emilia-mm-pretrain"


def _emilia_download_step() -> ExecutorStep[DownloadConfig]:
    """Use Google Storage Transfer Service for faster Emilia downloads."""
    return ExecutorStep(
        name=_EMILIA_DOWNLOAD_NAME,
        fn=storage_transfer_download,
        config=DownloadConfig(
            hf_dataset_id="potsawee/emilia-mm-pretrain",
            revision=EMILIA_REVISION,
            gcs_output_path=this_output_path(),
            hf_urls_glob=["Emilia*/**/*.parquet"],
            wait_for_completion=True,
        ),
    ).with_output_path(_EMILIA_DOWNLOAD_NAME)


def tokenize_emilia_steps() -> Mapping[str, ExecutorStep]:
    """Create tokenization steps for each Emilia split."""
    download_step = _emilia_download_step()

    steps: dict[str, ExecutorStep] = {_EMILIA_DOWNLOAD_NAME: download_step}
    for name, pattern in _EMILIA_SPLIT_PATTERNS:
        steps[name] = default_tokenize(
            name=name,
            dataset=download_step / pattern,
            tokenizer=QWEN3X_TOKENIZER,
            enforce_bos=False,
            enforce_eos=False,
        )
    return steps


# Yodas
_YODAS2_REVISION_STR = "8eda080a5fd6dfc070dd306c1e6446ab7c5b5f17"
YODAS2_REVISION = versioned(_YODAS2_REVISION_STR)
_YODAS2_DOWNLOAD_NAME = "raw/yodas2-mm-pretrain"
_YODAS2_SPLIT_PATTERNS: tuple[tuple[str, str], ...] = (("yodas2/en", "en{{000..007},{100..129}}/*.parquet"),)


def _yodas_download_step() -> ExecutorStep[DownloadConfig]:
    """Use Google Storage Transfer Service for faster YODAS2 downloads."""
    return ExecutorStep(
        name=_YODAS2_DOWNLOAD_NAME,
        fn=storage_transfer_download,
        config=DownloadConfig(
            hf_dataset_id="potsawee/yodas2-mm-pretrain",
            revision=YODAS2_REVISION,
            gcs_output_path=this_output_path(),
            hf_urls_glob=["**/*.parquet"],
            wait_for_completion=True,
        ),
    ).with_output_path(_YODAS2_DOWNLOAD_NAME)


def tokenize_yodas_steps() -> Mapping[str, ExecutorStep]:
    """Create tokenization steps for each YODAS2 language split."""
    download_step = _yodas_download_step()
    revision_root = (
        download_step / _YODAS2_REVISION_STR
    ) / f"huggingface.co/datasets/potsawee/yodas2-mm-pretrain/resolve/{_YODAS2_REVISION_STR}"
    steps: dict[str, ExecutorStep] = {_YODAS2_DOWNLOAD_NAME: download_step}
    for name, pattern in _YODAS2_SPLIT_PATTERNS:
        steps[name] = default_tokenize(
            name=name,
            dataset=revision_root / pattern,
            tokenizer=QWEN3X_TOKENIZER,
            enforce_bos=False,
            enforce_eos=False,
        )
    return steps


# MLS-EN
_MLS_EN_REVISION_STR = "69f96d72f6a46df18a3f91b72e3e0b42ff771460"
MLS_EN_REVISION = versioned(_MLS_EN_REVISION_STR)
_MLS_EN_DOWNLOAD_NAME = "raw/mls-en-mm-pretrain"


def _mls_en_download_step() -> ExecutorStep[DownloadConfig]:
    """Use Google Storage Transfer Service for faster MLS-EN downloads."""
    return ExecutorStep(
        name=_MLS_EN_DOWNLOAD_NAME,
        fn=storage_transfer_download,
        config=DownloadConfig(
            hf_dataset_id="potsawee/mls-en-mm-pretrain",
            revision=MLS_EN_REVISION,
            gcs_output_path=this_output_path(),
            hf_urls_glob=["data/*.parquet"],
            wait_for_completion=True,
        ),
    ).with_output_path(_MLS_EN_DOWNLOAD_NAME)


def tokenize_mls_en_steps() -> Mapping[str, ExecutorStep]:
    """Create tokenization steps for MLS-EN."""
    download_step = _mls_en_download_step()

    steps = {
        _MLS_EN_DOWNLOAD_NAME: download_step,
        "mls-en": default_tokenize(
            name="mls-en",
            dataset=download_step / "data/*.parquet",
            tokenizer=QWEN3X_TOKENIZER,
            enforce_bos=False,
            enforce_eos=False,
        ),
    }
    return steps


# Nemotron
nemotron_cc = nemotron_downloads["nemotron_cc"]
nemotron_cc_path = output_path_of(nemotron_cc, "contrib/Nemotron/Nemotron-CC/data-jsonl/")

# The following dataset splits define file patterns for each split.
NEMOTRON_DATASETS = {
    "hq_actual": ["quality=high/kind=actual/**/*.jsonl.gz"],
}


def _get_nemotron_split_paths(split):
    patterns = NEMOTRON_DATASETS[split]
    nemotron_split_paths = [nemotron_cc_path / pattern for pattern in patterns]
    return nemotron_split_paths


def tokenize_nemotron_hq_actual_step(*, base_path="tokenized/", tokenizer=QWEN3X_TOKENIZER) -> ExecutorStep:
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
    # if tokenizer in [llama3_tokenizer, YODAS2_TOKENIZER] and split in NEMOTRON_LLAMA3_OVERIDES:
    # step = step.with_output_path(NEMOTRON_LLAMA3_OVERIDES[split])
    return step


# 5.1 CVSS-method1 (speech to speech translation)
_CVSS_METHOD1_REVISION_STR = "90ca2216540ca74d97f49b68a2cc83b76cf3e296"
CVSS_METHOD1_REVISION = versioned(_CVSS_METHOD1_REVISION_STR)
_CVSS_METHOD1_SPLIT_PATTERN = "data/train*.parquet"
_CVSS_METHOD1_DOWNLOAD_NAME = "raw/cvss-mm-method1"


def _cvss_method1_download_step() -> ExecutorStep[DownloadConfig]:
    """Use Google Storage Transfer Service for faster CVSS-method1 downloads."""
    return ExecutorStep(
        name=_CVSS_METHOD1_DOWNLOAD_NAME,
        fn=storage_transfer_download,
        config=DownloadConfig(
            hf_dataset_id="potsawee/cvss-mm-method1",
            revision=CVSS_METHOD1_REVISION,
            gcs_output_path=this_output_path(),
            hf_urls_glob=["data/train*.parquet"],
            wait_for_completion=True,
        ),
    ).with_output_path(_CVSS_METHOD1_DOWNLOAD_NAME)


def tokenize_cvss_method1_steps() -> Mapping[str, ExecutorStep]:
    """Create tokenization steps for CVSS-method1 dataset."""
    download_step = _cvss_method1_download_step()
    steps: dict[str, ExecutorStep] = {_CVSS_METHOD1_DOWNLOAD_NAME: download_step}
    steps["cvss-method1"] = default_tokenize(
        name="cvss-method1",
        dataset=download_step / _CVSS_METHOD1_SPLIT_PATTERN,
        tokenizer=QWEN3X_TOKENIZER,
        enforce_bos=False,
        enforce_eos=False,
    )
    return steps


if __name__ == "__main__":
    steps = [
        # tokenize_emilia_steps()["Emilia/EN"],
        # tokenize_emilia_steps()["Emilia-YODAS/EN"],
        # tokenize_yodas_steps()["yodas2/en"],
        # tokenize_mls_en_steps()["mls-en"],
        # tokenize_nemotron_hq_actual_step(),
        *tokenize_cvss_method1_steps().values(),
    ]
    executor_main(steps=steps, description="Tokenize Qwen3x audio datasets")
