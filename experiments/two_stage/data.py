from marin.execution.executor import ExecutorStep, this_output_path
from operations.download.huggingface.download_hf import DownloadConfig, download_hf

from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from experiments.midtraining_datasets import finemath_3_plus_tokenized
from experiments.pretraining_datasets import slimpajama_6b
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer

dolma_components = tokenize_dolma_steps()

c4_tokenized = dolma_components["dolma/c4"]
starcoder_tokenized = dolma_components["dolma/starcoder"]
wiki_tokenized = dolma_components["dolma/wiki"]
flan_tokenized = dolma_components["dolma/flan"]

slimpajama_tokenized = default_tokenize(name="SlimPajama-6B", dataset=slimpajama_6b, tokenizer=llama3_tokenizer)

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

