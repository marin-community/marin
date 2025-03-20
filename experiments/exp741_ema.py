import dataclasses

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3_wrong
from experiments.defaults import default_train
from experiments.llama import llama_1_4b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main

llama_1_4b_train_config = SimpleTrainConfig(
    tpu_type="v4-128",
    node_count=1,
    train_batch_size=1024,
    num_train_steps=50_000,
    # these hypers from Table 12 in https://arxiv.org/html/2406.11794v1#A6
    learning_rate=1e-3,
    weight_decay=0.05,
    # WSD with EMA
    warmup=1000,
    decay=0.4,
    lr_schedule="linear",
    ema_beta=0.995,
)


llama_1b_tootsie = dataclasses.replace(
    default_train(
        name="llama-1b-ema",
        tokenized=dclm_mixture_config_llama3_wrong,
        model_config=llama_1_4b,
        train_config=llama_1_4b_train_config,
        tags=["llama", "1b", "ema", "exp741"],
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[
            llama_1b_tootsie,
        ],
        description="Train a 1B parameter model with EMA",
    )
