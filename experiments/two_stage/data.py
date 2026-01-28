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

from experiments.defaults import default_tokenize, default_download
from experiments.llama import llama3_tokenizer
from experiments.midtraining_datasets import finemath_3_plus_tokenized
from experiments.pretraining_datasets import tokenize_dolma
from experiments.pretraining_datasets.simple import tokenized
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path
from experiments.exp72_baselines import slimpajama_tokenized

dolma_components = tokenize_dolma()

c4_tokenized = dolma_components["dolma/c4"]
starcoder_tokenized = dolma_components["dolma/starcoder"]
wiki_tokenized = dolma_components["dolma/wiki"]
flan_tokenized = dolma_components["dolma/flan"]

slimpajama_6b_tokenized = tokenized["slimpajama_6b"]

latxa_corpus = ExecutorStep(
    name="raw/latxa_corpus",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="HiTZ/latxa-corpus-v1.1",
        revision="02dc515",
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

latxa_corpus_tokenized = default_tokenize(
    name="latxa_corpus",
    dataset=latxa_corpus,
    tokenizer=llama3_tokenizer,
)

octothinker_megamath = default_download(
    name="raw/octothinker-megamath",
    hf_dataset_id="OctoThinker/MegaMath-Web-Pro-Max",
    revision="b5129b6",
    override_output_path="raw/octothinker-megamath",
)

octothinker_megamath_tokenized = default_tokenize(
    name="octothinker-megamath",
    dataset=octothinker_megamath,
    tokenizer=llama3_tokenizer,
)

data_dict = {
    "c4": c4_tokenized,
    "spj": slimpajama_6b_tokenized,
    "spj_full": slimpajama_tokenized,
    "starcoder": starcoder_tokenized,
    "wiki": wiki_tokenized,
    "flan": flan_tokenized,
    "finemath": finemath_3_plus_tokenized,
    "latxa": latxa_corpus_tokenized,
    "octo": octothinker_megamath_tokenized,
}
