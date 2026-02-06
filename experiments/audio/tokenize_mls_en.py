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

"""Tokenize full MLS-EN dataset for Marin audio experiments."""

from collections.abc import Mapping

from levanter.data.text import LMMixtureDatasetConfig

from experiments.defaults import default_tokenize
from marin.download.huggingface.download_hf import DownloadConfig, download_hf as storage_transfer_download
from marin.execution import executor_main, versioned
from marin.execution.executor import ExecutorStep, this_output_path
from marin.processing.tokenize.data_configs import lm_data_config

_MLS_EN_REVISION_STR = "69f96d72f6a46df18a3f91b72e3e0b42ff771460"
MLS_EN_REVISION = versioned(_MLS_EN_REVISION_STR)
MLS_EN_TOKENIZER = "potsawee/marin-mimi-bpe-8cb-16k-tokenizer"

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
            tokenizer=MLS_EN_TOKENIZER,
            enforce_bos=False,
            enforce_eos=False,
        ),
    }
    return steps


def mls_en_data_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized MLS-EN."""
    steps = tokenize_mls_en_steps()
    training_set = steps["mls-en"]
    return lm_data_config(training_set=training_set)


# MLS-EN TTS0 (as TTS only dataset)

_MLS_EN_TTS0_REVISION_STR = "66768dda6f495fd731ad2c79ee00363e2d81ddaf"
MLS_EN_TTS0_REVISION = versioned(_MLS_EN_TTS0_REVISION_STR)
_MLS_EN_TTS0_DOWNLOAD_NAME = "raw/mls-en-mm-tts0"


def _mls_en_tts0_download_step() -> ExecutorStep[DownloadConfig]:
    """Use Google Storage Transfer Service for faster MLS-EN TTS0 downloads."""
    return ExecutorStep(
        name=_MLS_EN_TTS0_DOWNLOAD_NAME,
        fn=storage_transfer_download,
        config=DownloadConfig(
            hf_dataset_id="potsawee/mls-en-mm-tts0",
            revision=MLS_EN_TTS0_REVISION,
            gcs_output_path=this_output_path(),
            hf_urls_glob=["data/*.parquet"],
            wait_for_completion=True,
        ),
    ).with_output_path(_MLS_EN_TTS0_DOWNLOAD_NAME)


def tokenize_mls_en_tts0_steps() -> Mapping[str, ExecutorStep]:
    """Create tokenization steps for MLS-EN TTS0."""
    download_step = _mls_en_tts0_download_step()

    steps = {
        _MLS_EN_TTS0_DOWNLOAD_NAME: download_step,
        "mls-en-tts0": default_tokenize(
            name="mls-en-tts0",
            dataset=download_step / "data/*.parquet",
            tokenizer=MLS_EN_TOKENIZER,
            enforce_bos=False,
            enforce_eos=False,
        ),
    }
    return steps


if __name__ == "__main__":
    # executor_main(steps=list(tokenize_mls_en_steps().values()))
    executor_main(steps=list(tokenize_mls_en_tts0_steps().values()))
