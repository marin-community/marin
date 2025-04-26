"""
Train 1.4B models on Fineweb Edu with different tokenizers.
https://github.com/stanford-crfm/marin/issues/524
"""

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama_1_4b, llama_1_4b_train_config
from experiments.pretraining_datasets import fineweb_edu
from marin.execution.executor import executor_main

fineweb_edu_llama3_tokenized = default_tokenize(
    name="fineweb-edu", dataset=fineweb_edu, tokenizer="meta-llama/Meta-Llama-3.1-8B"
)
llama_1_4b_fineweb_edu_llama3_model = default_train(
    name="llama3-tokenizer",
    tokenized=fineweb_edu_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

fineweb_edu_llama2_tokenized = default_tokenize(
    name="fineweb-edu", dataset=fineweb_edu, tokenizer="meta-llama/Llama-2-7b-hf"
)
llama_1_4b_fineweb_edu_llama2_model = default_train(
    name="llama2-tokenizer",
    tokenized=fineweb_edu_llama2_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

fineweb_edu_neox_tokenized = default_tokenize(
    name="fineweb-edu", dataset=fineweb_edu, tokenizer="EleutherAI/gpt-neox-20b"
)
llama_1_4b_fineweb_edu_neox_model = default_train(
    name="neox-tokenizer",
    tokenized=fineweb_edu_neox_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)


############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            llama_1_4b_fineweb_edu_llama3_model,
            llama_1_4b_fineweb_edu_llama2_model,
            llama_1_4b_fineweb_edu_neox_model,
        ],
        description="Train 1.4B models on Fineweb Edu with three different tokenizers (llama3, llama3, neox).",
    )
