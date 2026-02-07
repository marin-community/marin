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

"""Tokenize various datasets for cooling down experiments."""

from collections.abc import Mapping

from experiments.defaults import default_tokenize
from marin.download.huggingface.download_hf import DownloadConfig, download_hf as storage_transfer_download
from marin.execution import executor_main, versioned
from marin.execution.executor import ExecutorStep, this_output_path

TOKENIZER = "potsawee/marin-mimi-bpe-8cb-16k-tokenizer"

# 1. Peoples Speech
_PEOPLES_SPEECH_REVISION_STR = "b3cdef3771a994bb09f38a14c42354f480c1f0ac"
PEOPLES_SPEECH_REVISION = versioned(_PEOPLES_SPEECH_REVISION_STR)
_PEOPLES_SPEECH_SPLIT_PATTERN = "clean*/*.parquet"
_PEOPLES_SPEECH_DOWNLOAD_NAME = "raw/peoples-speech-mm-pretrain"


def _peoples_speech_download_step() -> ExecutorStep[DownloadConfig]:
    """Use Google Storage Transfer Service for faster Peoples Speech downloads."""
    return ExecutorStep(
        name=_PEOPLES_SPEECH_DOWNLOAD_NAME,
        fn=storage_transfer_download,
        config=DownloadConfig(
            hf_dataset_id="potsawee/peoples-speech-mm-pretrain",
            revision=PEOPLES_SPEECH_REVISION,
            gcs_output_path=this_output_path(),
            hf_urls_glob=["clean*/*.parquet"],
            wait_for_completion=True,
        ),
    ).with_output_path(_PEOPLES_SPEECH_DOWNLOAD_NAME)


def tokenize_peoples_speech_steps() -> Mapping[str, ExecutorStep]:
    """Create tokenization steps for Peoples Speech dataset."""
    download_step = _peoples_speech_download_step()
    steps: dict[str, ExecutorStep] = {_PEOPLES_SPEECH_DOWNLOAD_NAME: download_step}
    steps["peoples-speech-clean"] = default_tokenize(
        name="peoples-speech-clean",
        dataset=download_step / _PEOPLES_SPEECH_SPLIT_PATTERN,
        tokenizer=TOKENIZER,
        enforce_bos=False,
        enforce_eos=False,
    )
    return steps


# 2. Common Voice 17 (English only)
_COMMON_VOICE_17_REVISION_STR = "c30dfc4fb604fa9bdc90db470ce510e0ec2fa4a4"
COMMON_VOICE_17_REVISION = versioned(_COMMON_VOICE_17_REVISION_STR)
_COMMON_VOICE_17_SPLIT_PATTERN = "en/*.parquet"
_COMMON_VOICE_17_DOWNLOAD_NAME = "raw/commonvoice17-mm-pretrain"


def _common_voice_17_download_step() -> ExecutorStep[DownloadConfig]:
    """Use Google Storage Transfer Service for faster Common Voice 17 downloads."""
    return ExecutorStep(
        name=_COMMON_VOICE_17_DOWNLOAD_NAME,
        fn=storage_transfer_download,
        config=DownloadConfig(
            hf_dataset_id="potsawee/commonvoice17-mm-pretrain",
            revision=COMMON_VOICE_17_REVISION,
            gcs_output_path=this_output_path(),
            hf_urls_glob=["en/*.parquet"],
            wait_for_completion=True,
        ),
    ).with_output_path(_COMMON_VOICE_17_DOWNLOAD_NAME)


def tokenize_common_voice_17_steps() -> Mapping[str, ExecutorStep]:
    """Create tokenization steps for Common Voice 17 dataset."""
    download_step = _common_voice_17_download_step()
    steps: dict[str, ExecutorStep] = {_COMMON_VOICE_17_DOWNLOAD_NAME: download_step}
    steps["commonvoice17-en"] = default_tokenize(
        name="commonvoice17-en",
        dataset=download_step / _COMMON_VOICE_17_SPLIT_PATTERN,
        tokenizer=TOKENIZER,
        enforce_bos=False,
        enforce_eos=False,
    )
    return steps


# 3. LibriSpeech
_LIBRISPEECH_REVISION_STR = "f9a47e5e725ae3298bb4e8f1e8979d068044e3ab"
LIBRISPEECH_REVISION = versioned(_LIBRISPEECH_REVISION_STR)
_LIBRISPEECH_SPLIT_PATTERN = "data/train*.parquet"
_LIBRISPEECH_DOWNLOAD_NAME = "raw/librispeech-mm-pretrain"


def _librispeech_download_step() -> ExecutorStep[DownloadConfig]:
    """Use Google Storage Transfer Service for faster LibriSpeech downloads."""
    return ExecutorStep(
        name=_LIBRISPEECH_DOWNLOAD_NAME,
        fn=storage_transfer_download,
        config=DownloadConfig(
            hf_dataset_id="potsawee/librispeech-mm-pretrain",
            revision=LIBRISPEECH_REVISION,
            gcs_output_path=this_output_path(),
            hf_urls_glob=["data/train*.parquet"],
            wait_for_completion=True,
        ),
    ).with_output_path(_LIBRISPEECH_DOWNLOAD_NAME)


def tokenize_librispeech_steps() -> Mapping[str, ExecutorStep]:
    """Create tokenization steps for LibriSpeech dataset."""
    download_step = _librispeech_download_step()
    steps: dict[str, ExecutorStep] = {_LIBRISPEECH_DOWNLOAD_NAME: download_step}
    steps["librispeech-train"] = default_tokenize(
        name="librispeech-train",
        dataset=download_step / _LIBRISPEECH_SPLIT_PATTERN,
        tokenizer=TOKENIZER,
        enforce_bos=False,
        enforce_eos=False,
    )
    return steps


# 4. LibriTTS
_LIBRITTS_REVISION_STR = "ca8b1c7022f3783ec41e11f5a96ac448dc36d2a2"
LIBRITTS_REVISION = versioned(_LIBRITTS_REVISION_STR)
_LIBRITTS_SPLIT_PATTERN = "data/train*.parquet"
_LIBRITTS_DOWNLOAD_NAME = "raw/libritts-r-mm-pretrain"


def _libritts_download_step() -> ExecutorStep[DownloadConfig]:
    """Use Google Storage Transfer Service for faster LibriTTS downloads."""
    return ExecutorStep(
        name=_LIBRITTS_DOWNLOAD_NAME,
        fn=storage_transfer_download,
        config=DownloadConfig(
            hf_dataset_id="potsawee/libritts-r-mm-pretrain",
            revision=LIBRITTS_REVISION,
            gcs_output_path=this_output_path(),
            hf_urls_glob=["data/train*.parquet"],
            wait_for_completion=True,
        ),
    ).with_output_path(_LIBRITTS_DOWNLOAD_NAME)


def tokenize_libritts_steps() -> Mapping[str, ExecutorStep]:
    """Create tokenization steps for LibriTTS dataset."""
    download_step = _libritts_download_step()
    steps: dict[str, ExecutorStep] = {_LIBRITTS_DOWNLOAD_NAME: download_step}
    steps["libritts-train"] = default_tokenize(
        name="libritts-train",
        dataset=download_step / _LIBRITTS_SPLIT_PATTERN,
        tokenizer=TOKENIZER,
        enforce_bos=False,
        enforce_eos=False,
    )
    return steps


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
        tokenizer=TOKENIZER,
        enforce_bos=False,
        enforce_eos=False,
    )
    return steps


if __name__ == "__main__":
    steps = [
        *tokenize_peoples_speech_steps().values(),
        *tokenize_common_voice_17_steps().values(),
        *tokenize_librispeech_steps().values(),
        *tokenize_libritts_steps().values(),
        *tokenize_cvss_method1_steps().values(),
    ]
    executor_main(steps=steps)
