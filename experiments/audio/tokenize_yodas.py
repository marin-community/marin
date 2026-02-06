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

"""Tokenize each YODAS2 language split for Marin audio experiments."""

from collections.abc import Mapping

from levanter.data.text import LMMixtureDatasetConfig

from experiments.defaults import default_tokenize
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution import executor_main, versioned
from marin.execution.executor import ExecutorStep, this_output_path
from marin.processing.tokenize.data_configs import TokenizerStep, lm_mixture_data_config, lm_data_config

_YODAS2_LANGUAGE_WEIGHTS = {
    "en": 0.355727909201,
    "th": 0.000970575517,
    "ar": 0.00059,
    "de": 0.023203517508,
    "es": 0.073786617338,
    "fr": 0.034971397976,
    "hi": 0.000784368275,
    "zh": 0.00050167398,
}

_YODAS2_REVISION_STR = "8eda080a5fd6dfc070dd306c1e6446ab7c5b5f17"
YODAS2_REVISION = versioned(_YODAS2_REVISION_STR)
YODAS2_TOKENIZER = "potsawee/marin-mimi-bpe-8cb-16k-tokenizer"
YODAS2_QWEN_TOKENIZER = "potsawee/qwen3-mimi-bpe-8cb-16k-tokenizer"

_YODAS2_SPLIT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("yodas2/en", "en{{000..007},{100..129}}/*.parquet"),
    ("yodas2/th", "th{000,100}/*.parquet"),
    ("yodas2/ar", "ar000/*.parquet"),
    ("yodas2/de", "de{000,{100..102}}/*.parquet"),
    ("yodas2/es", "es{000,{100..108}}/*.parquet"),
    ("yodas2/fr", "fr{000,{100..103}}/*.parquet"),
    ("yodas2/hi", "hi{000,100}/*.parquet"),
    ("yodas2/zh", "zh000/*.parquet"),
)

_YODAS2_DOWNLOAD_NAME = "raw/yodas2-mm-pretrain"


def _yodas_download_step() -> ExecutorStep[DownloadConfig]:
    """Use Google Storage Transfer Service for faster YODAS2 downloads."""
    return ExecutorStep(
        name=_YODAS2_DOWNLOAD_NAME,
        fn=download_hf,
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
    ) / "huggingface.co/datasets/potsawee/yodas2-mm-pretrain/resolve/8eda080a5fd6dfc070dd306c1e6446ab7c5b5f17"

    steps: dict[str, ExecutorStep] = {_YODAS2_DOWNLOAD_NAME: download_step}
    for name, pattern in _YODAS2_SPLIT_PATTERNS:
        steps[name] = default_tokenize(
            name=name,
            dataset=revision_root / pattern,
            tokenizer=YODAS2_TOKENIZER,
            enforce_bos=False,
            enforce_eos=False,
        )
    return steps


def tokenize_yodas_for_qwen3_steps() -> Mapping[str, ExecutorStep]:
    """Create tokenization steps for each YODAS2 language split."""
    download_step = _yodas_download_step()
    revision_root = (
        download_step / _YODAS2_REVISION_STR
    ) / "huggingface.co/datasets/potsawee/yodas2-mm-pretrain/resolve/8eda080a5fd6dfc070dd306c1e6446ab7c5b5f17"

    steps: dict[str, ExecutorStep] = {_YODAS2_DOWNLOAD_NAME: download_step}
    for name, pattern in _YODAS2_SPLIT_PATTERNS:
        steps[name] = default_tokenize(
            name=name,
            dataset=revision_root / pattern,
            tokenizer=YODAS2_QWEN_TOKENIZER,
            enforce_bos=False,
            enforce_eos=False,
        )
    return steps


def yodas2_tokenized_steps() -> dict[str, TokenizerStep]:
    """Return only the tokenization steps for the YODAS2 language splits."""
    steps = tokenize_yodas_steps()
    language_names = {f"yodas2/{lang}" for lang in _YODAS2_LANGUAGE_WEIGHTS}
    return {name: step for name, step in steps.items() if name in language_names}


def yodas2_qwen3_tokenized_steps() -> dict[str, TokenizerStep]:
    """Return only the tokenization steps (for Qwen3) for the YODAS2 language splits."""
    steps = tokenize_yodas_for_qwen3_steps()
    language_names = {f"yodas2/{lang}" for lang in _YODAS2_LANGUAGE_WEIGHTS}
    return {name: step for name, step in steps.items() if name in language_names}


def yodas2_mixture_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized YODAS2 splits."""
    tokenized = yodas2_tokenized_steps()
    weights = {f"yodas2/{lang}": weight for lang, weight in _YODAS2_LANGUAGE_WEIGHTS.items()}
    return lm_mixture_data_config(
        components=tokenized,
        weights=weights,
        permutation_type="feistel",
    )


def yodas2_english_data_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized YODAS2 English only."""
    tokenized = yodas2_tokenized_steps()
    training_set = tokenized["yodas2/en"]
    return lm_data_config(training_set=training_set)


def yodas2_qwen3_mixture_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized YODAS2 splits (for Qwen3)."""
    tokenized = yodas2_qwen3_tokenized_steps()
    weights = {f"yodas2/{lang}": weight for lang, weight in _YODAS2_LANGUAGE_WEIGHTS.items()}
    return lm_mixture_data_config(
        components=tokenized,
        weights=weights,
        permutation_type="feistel",
    )


def yodas2_qwen3_english_data_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized YODAS2 English only (for Qwen3)."""
    tokenized = yodas2_qwen3_tokenized_steps()
    training_set = tokenized["yodas2/en"]
    return lm_data_config(training_set=training_set)


# YODAS2 as ASR only dataset

_YODAS2_ASR_REVISION_STR = "ab549c2f0b27a767ec6fb59f66dd5f8932d1fb40"
YODAS2_ASR_REVISION = versioned(_YODAS2_ASR_REVISION_STR)
_YODAS2_ASR_DOWNLOAD_NAME = "raw/yodas2-mm-asr"


def _yodas_asr_en_download_step() -> ExecutorStep[DownloadConfig]:
    """Use Google Storage Transfer Service for faster YODAS2 downloads."""
    return ExecutorStep(
        name=_YODAS2_ASR_DOWNLOAD_NAME,
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="potsawee/yodas2-mm-asr",
            revision=YODAS2_ASR_REVISION,
            gcs_output_path=this_output_path(),
            hf_urls_glob=["en*/*.parquet"],
            wait_for_completion=True,
        ),
    ).with_output_path(_YODAS2_ASR_DOWNLOAD_NAME)


def tokenize_yodas_asr_en_steps() -> Mapping[str, ExecutorStep]:
    """Create tokenization steps for each YODAS2 language split."""
    download_step = _yodas_asr_en_download_step()
    steps: dict[str, ExecutorStep] = {_YODAS2_ASR_DOWNLOAD_NAME: download_step}
    steps["yodas2/en"] = default_tokenize(
        name="yodas2/en",
        dataset=download_step / "en*/*.parquet",
        tokenizer=YODAS2_TOKENIZER,
        enforce_bos=False,
        enforce_eos=False,
    )
    return steps


# YODAS2 for Acoustic-only and Semantic-only

## Acoustic-only
_YODAS2_ACOUSTIC_REVISION_STR = "a92603918d6e858ffa4e6ad7581199828f63c286"
YODAS2_ACOUSTIC_REVISION = versioned(_YODAS2_ACOUSTIC_REVISION_STR)
_YODAS2_ACOUSTIC_DOWNLOAD_NAME = "raw/yodas2-mm-acoustic"


def _yodas_acoustic_en_download_step() -> ExecutorStep[DownloadConfig]:
    """Use Google Storage Transfer Service for faster YODAS2 downloads."""
    return ExecutorStep(
        name=_YODAS2_ACOUSTIC_DOWNLOAD_NAME,
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="potsawee/yodas2-mm-acoustic",
            revision=YODAS2_ACOUSTIC_REVISION,
            gcs_output_path=this_output_path(),
            hf_urls_glob=["en*/*.parquet"],
            wait_for_completion=True,
        ),
    ).with_output_path(_YODAS2_ACOUSTIC_DOWNLOAD_NAME)


def tokenize_yodas_acoustic_en_steps() -> Mapping[str, ExecutorStep]:
    """Create tokenization steps for each YODAS2 language split."""
    download_step = _yodas_acoustic_en_download_step()
    steps: dict[str, ExecutorStep] = {_YODAS2_ACOUSTIC_DOWNLOAD_NAME: download_step}
    steps["yodas2/en"] = default_tokenize(
        name="yodas2/en",
        dataset=download_step / "en*/*.parquet",
        tokenizer=YODAS2_TOKENIZER,
        enforce_bos=False,
        enforce_eos=False,
    )
    return steps


def yodas2_acoustic_english_data_config() -> LMMixtureDatasetConfig:
    """Create an lm_data_config for the tokenized YODAS2 English only (Acoustic-only)."""
    tokenized = tokenize_yodas_acoustic_en_steps()
    training_set = tokenized["yodas2/en"]
    return lm_data_config(training_set=training_set)


## Semantic-only
_YODAS2_SEMANTIC_REVISION_STR = "9912c02b9804c525e88c4cdd77cd5674d117f11f"
YODAS2_SEMANTIC_REVISION = versioned(_YODAS2_SEMANTIC_REVISION_STR)
_YODAS2_SEMANTIC_DOWNLOAD_NAME = "raw/yodas2-mm-semantic"


def _yodas_semantic_en_download_step() -> ExecutorStep[DownloadConfig]:
    """Use Google Storage Transfer Service for faster YODAS2 downloads."""
    return ExecutorStep(
        name=_YODAS2_SEMANTIC_DOWNLOAD_NAME,
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="potsawee/yodas2-mm-semantic",
            revision=YODAS2_SEMANTIC_REVISION,
            gcs_output_path=this_output_path(),
            hf_urls_glob=["en*/*.parquet"],
            wait_for_completion=True,
        ),
    ).with_output_path(_YODAS2_SEMANTIC_DOWNLOAD_NAME)


def tokenize_yodas_semantic_en_steps() -> Mapping[str, ExecutorStep]:
    """Create tokenization steps for each YODAS2 language split."""
    download_step = _yodas_semantic_en_download_step()
    steps: dict[str, ExecutorStep] = {_YODAS2_SEMANTIC_DOWNLOAD_NAME: download_step}
    steps["yodas2/en"] = default_tokenize(
        name="yodas2/en",
        dataset=download_step / "en*/*.parquet",
        tokenizer=YODAS2_TOKENIZER,
        enforce_bos=False,
        enforce_eos=False,
    )
    return steps


def yodas2_semantic_english_data_config() -> LMMixtureDatasetConfig:
    """Create an lm_data_config for the tokenized YODAS2 English only (Semantic-only)."""
    tokenized = tokenize_yodas_semantic_en_steps()
    training_set = tokenized["yodas2/en"]
    return lm_data_config(training_set=training_set)


if __name__ == "__main__":
    # executor_main(steps=list(tokenize_yodas_steps().values()))
    # executor_main(steps=list(tokenize_yodas_for_qwen3_steps().values()))
    # executor_main(steps=list(tokenize_yodas_asr_en_steps().values()))

    # acoustic only and semantic only
    steps = []
    for step in tokenize_yodas_acoustic_en_steps().values():
        steps.append(step)
    for step in tokenize_yodas_semantic_en_steps().values():
        steps.append(step)
    executor_main(steps=steps)
