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

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from experiments.midtraining_datasets import tokenize_finemath_3_plus
from experiments.pretraining_datasets import tokenize_dolma
from experiments.pretraining_datasets.simple import tokenized
from marin.download.huggingface.download_hf import DownloadConfig
from marin.download.huggingface.download_hf import download_hf as _download_hf
from marin.execution import step, deferred, output

download_hf = deferred(_download_hf)


def get_dolma_components():
    return tokenize_dolma()


def get_c4_tokenized():
    return get_dolma_components()["dolma/c4"]


def get_starcoder_tokenized():
    return get_dolma_components()["dolma/starcoder"]


def get_wiki_tokenized():
    return get_dolma_components()["dolma/wiki"]


def get_flan_tokenized():
    return get_dolma_components()["dolma/flan"]


def get_slimpajama_tokenized():
    return tokenized["slimpajama_6b"]


@step(name="raw/latxa_corpus")
def latxa_corpus():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="HiTZ/latxa-corpus-v1.1",
            revision="02dc515",
            gcs_output_path=output(),
            wait_for_completion=True,
        )
    )


def tokenize_latxa_corpus():
    return default_tokenize(
        name="latxa_corpus",
        dataset=latxa_corpus(),
        tokenizer=llama3_tokenizer,
    )


class DataDict:
    """Lazy dictionary that creates tokenized data steps on demand."""

    def __getitem__(self, key: str):
        if key == "c4":
            return get_c4_tokenized()
        elif key == "spj":
            return get_slimpajama_tokenized()
        elif key == "starcoder":
            return get_starcoder_tokenized()
        elif key == "wiki":
            return get_wiki_tokenized()
        elif key == "flan":
            return get_flan_tokenized()
        elif key == "finemath":
            return tokenize_finemath_3_plus()
        elif key == "latxa":
            return tokenize_latxa_corpus()
        else:
            raise KeyError(f"Unknown data key: {key}")


data_dict = DataDict()
