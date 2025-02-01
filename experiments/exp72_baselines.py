"""
Train 1.4B models on standard datasets (e.g., SlimPajama).
https://github.com/stanford-crfm/marin/issues/72
"""

import os

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config, llama_300m, llama_300m_train_config
from experiments.pretraining_datasets import fineweb_edu, nemotron_cc, slimpajama, slimpajama_6b
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

slimpajama_6b_tokenized = default_tokenize(name="SlimPajama-6B", dataset=slimpajama_6b, tokenizer=llama3_tokenizer)
slimpajama_6b_model = default_train(
    name="SlimPajama-6B-300m",
    tokenized=slimpajama_6b_tokenized,
    model_config=llama_300m,
    train_config=llama_300m_train_config,
)

slimpajama_tokenized = ExecutorStep(
    name=os.path.join("tokenized", "SlimPajama-627B"),
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[slimpajama.cd("train")],
        validation_paths=[slimpajama.cd("validation")],
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
    ),
)
slimpajama_model = default_train(
    name="SlimPajama-627B-1.4b",
    tokenized=slimpajama_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

fineweb_edu_tokenized = default_tokenize(name="fineweb-edu", dataset=fineweb_edu, tokenizer=llama3_tokenizer)
fineweb_edu_model = default_train(
    name="fineweb-edu-1.4b",
    tokenized=fineweb_edu_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

nemotron_cc_tokenized = default_tokenize(name="nemotron_cc", dataset=nemotron_cc, tokenizer=llama3_tokenizer)
nemotron_cc_model = default_train(
    name="nemotron_cc-1.4b",
    tokenized=nemotron_cc_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            slimpajama_6b_model,
            slimpajama_model,
            fineweb_edu_model,
            nemotron_cc_tokenized,
            nemotron_cc_model,
        ],
        description="Train 1.4B models on standard datasets (SlimPajama 6B, SlimPajama, FineWebEdu, Nemotron-CC).",
    )
