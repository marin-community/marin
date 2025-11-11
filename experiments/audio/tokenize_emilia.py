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

"""Tokenize each Emilia split for Marin audio experiments."""

from collections.abc import Mapping

from levanter.data.text import LMMixtureDatasetConfig

from experiments.defaults import default_tokenize
from marin.download.huggingface.download import DownloadConfig, download as storage_transfer_download
from marin.execution import executor_main, versioned
from marin.execution.executor import ExecutorStep, this_output_path
from marin.processing.tokenize.data_configs import TokenizerStep, lm_mixture_data_config

_EMILIA_WEIGHTS = {
    "Emilia/EN": 0.217169374,
    "Emilia/ZH": 0.231554524,
    "Emilia/DE": 0.007424594,
    "Emilia/FR": 0.006496520,
    "Emilia/JA": 0.007888631,
    "Emilia/KO": 0.000928074,
    "Emilia-YODAS/EN": 0.427842227,
    "Emilia-YODAS/ZH": 0.001392111,
    "Emilia-YODAS/DE": 0.025986079,
    "Emilia-YODAS/FR": 0.034338747,
    "Emilia-YODAS/JA": 0.005104408,
    "Emilia-YODAS/KO": 0.033874710,
}

_EMILIA_REVISION_STR = "e41392e668b273145344fda2f737cd75a6ebb558"
EMILIA_REVISION = versioned(_EMILIA_REVISION_STR)
EMILIA_TOKENIZER = "potsawee/marin-mimi-bpe-8cb-16k-tokenizer"

_EMILIA_SPLIT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Emilia/EN", "Emilia/EN/*.parquet"),
    ("Emilia/ZH", "Emilia/ZH/*.parquet"),
    ("Emilia/DE", "Emilia/DE/*.parquet"),
    ("Emilia/FR", "Emilia/FR/*.parquet"),
    ("Emilia/JA", "Emilia/JA/*.parquet"),
    ("Emilia/KO", "Emilia/KO/*.parquet"),
    ("Emilia-YODAS/EN", "Emilia-YODAS/EN/*.parquet"),
    ("Emilia-YODAS/ZH", "Emilia-YODAS/ZH/*.parquet"),
    ("Emilia-YODAS/DE", "Emilia-YODAS/DE/*.parquet"),
    ("Emilia-YODAS/FR", "Emilia-YODAS/FR/*.parquet"),
    ("Emilia-YODAS/JA", "Emilia-YODAS/JA/*.parquet"),
    ("Emilia-YODAS/KO", "Emilia-YODAS/KO/*.parquet"),
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
    revision_root = (
        download_step / _EMILIA_REVISION_STR
    ) / "huggingface.co/datasets/potsawee/emilia-mm-pretrain/resolve/e41392e668b273145344fda2f737cd75a6ebb558"

    steps: dict[str, ExecutorStep] = {_EMILIA_DOWNLOAD_NAME: download_step}
    for name, pattern in _EMILIA_SPLIT_PATTERNS:
        steps[name] = default_tokenize(
            name=name,
            dataset=revision_root / pattern,
            tokenizer=EMILIA_TOKENIZER,
            enforce_bos=False,
            enforce_eos=False,
        )
    return steps


def emilia_tokenized_steps() -> dict[str, TokenizerStep]:
    """Return only the tokenization steps for the Emilia splits."""
    steps = tokenize_emilia_steps()
    split_names = {split for split in _EMILIA_WEIGHTS}
    return {name: step for name, step in steps.items() if name in split_names}


def emilia_mixture_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized Emilia splits."""
    tokenized = emilia_tokenized_steps()
    weights = {split: weight for split, weight in _EMILIA_WEIGHTS.items()}
    return lm_mixture_data_config(
        components=tokenized,
        weights=weights,
        permutation_type="feistel",
    )


def emilia_english_mixture_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized Emilia English splits."""
    tokenized = emilia_tokenized_steps()
    tokenized_english = {name: step for name, step in tokenized.items() if name in ["Emilia/EN", "Emilia-YODAS/EN"]}
    weights = {
        "Emilia/EN": _EMILIA_WEIGHTS["Emilia/EN"],
        "Emilia-YODAS/EN": _EMILIA_WEIGHTS["Emilia-YODAS/EN"],
    }
    return lm_mixture_data_config(
        components=tokenized_english,
        weights=weights,
        permutation_type="feistel",
    )


if __name__ == "__main__":
    executor_main(steps=list(tokenize_emilia_steps().values()))
