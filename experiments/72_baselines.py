"""
Train 1.4B models on standard datasets.
https://github.com/stanford-crfm/marin/issues/72
"""
import dataclasses
import os

from levanter.store.cache import CacheOptions

from experiments.defaults import SimpleTrainConfig, default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_150m
from experiments.pretraining_datasets import fineweb_edu, slimpajama, slimpajama_6b
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

# TODO: tune these hyperparameters
default_train_config = SimpleTrainConfig(
    tpu_type="v4-8",
    train_batch_size=64,
    num_train_steps=100,
    learning_rate=4e-4,
    weight_decay=0.1,
)

# Just a small test
hello_world_fw = "gs://marin-us-central2/documents/hello_world_fw/v1.0/quickstart"
hello_world_fw_tokenized = default_tokenize(name="hello_world_fw", dataset=hello_world_fw, tokenizer=llama3_tokenizer)
hello_world_fw_model = default_train(
    name="hello_world_fw",
    tokenized=hello_world_fw_tokenized,
    model_config=llama_150m,
    train_config=default_train_config,
)

slimpajama_6b_tokenized = default_tokenize(name="SlimPajama-6B", dataset=slimpajama_6b, tokenizer=llama3_tokenizer)
slimpajama_6b_model = default_train(
    name="SlimPajama-6B-1.4b",
    tokenized=slimpajama_6b_tokenized,
    model_config=llama_1_4b,
    train_config=default_train_config,
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

slimpajama_6b_model = default_train(
    name="SlimPajama-6B-1.4b",
    tokenized=slimpajama_6b_tokenized,
    model_config=llama_1_4b,
    train_config=default_train_config,
)

fineweb_edu_tokenized = default_tokenize(name="fineweb-edu", dataset=fineweb_edu, tokenizer=llama3_tokenizer)
fineweb_edu_model = default_train(
    name="fineweb-edu-1.4b",
    tokenized=fineweb_edu_tokenized,
    model_config=llama_1_4b,
    train_config=default_train_config,
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            # slimpajama,
            # fineweb_edu,
            # hello_world_fw_model,
            slimpajama_tokenized,
            slimpajama_6b_model
        ]
    )
