"""
Train 1.4B models on standard datasets (e.g., SlimPajama) using multislice.
https://github.com/stanford-crfm/marin/issues/146
"""

from experiments.defaults import default_tokenize, default_train
from experiments.llama import (
    llama3_tokenizer,
    llama_1_4b,
    llama_1_4b_multislice_train_config,
)
from experiments.pretraining_datasets import fineweb_edu
from marin.execution.executor import executor_main

fineweb_edu_tokenized = default_tokenize(name="fineweb-edu", dataset=fineweb_edu, tokenizer=llama3_tokenizer)
fineweb_edu_model = default_train(
    name="fineweb-edu-1.4b-multislice",
    tokenized=fineweb_edu_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_multislice_train_config,
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[fineweb_edu_model],
        description="Train 1.4B models on standard datasets (SlimPajama 6B, SlimPajama, FineWebEdu).",
    )
