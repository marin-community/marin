# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from experiments.midtraining_datasets import finemath_3_plus_tokenized
from experiments.pretraining_datasets import tokenize_dolma
from experiments.pretraining_datasets.simple import tokenized
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path

dolma_components = tokenize_dolma()

c4_tokenized = dolma_components["dolma/c4"]
starcoder_tokenized = dolma_components["dolma/starcoder"]
wiki_tokenized = dolma_components["dolma/wiki"]
flan_tokenized = dolma_components["dolma/flan"]

slimpajama_tokenized = tokenized["slimpajama_6b"]

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

data_dict = {
    "c4": c4_tokenized,
    "spj": slimpajama_tokenized,
    "starcoder": starcoder_tokenized,
    "wiki": wiki_tokenized,
    "flan": flan_tokenized,
    "finemath": finemath_3_plus_tokenized,
    "latxa": latxa_corpus_tokenized,
}
