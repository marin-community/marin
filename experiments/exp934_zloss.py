"""
Experiment 934: See if we should make zloss be default on
"""

import dataclasses

from experiments.defaults import default_train
from experiments.exp72_baselines import nemotron_cc_model, nemotron_cc_tokenized
from experiments.llama import llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import executor_main

llama_1_4b_train_config_wsd = dataclasses.replace(
    llama_1_4b_train_config,
    weight_decay=0.05,
    decay=0.2,
    lr_schedule="linear",
    ema_beta=0.995,
)


llama_1_4b_train_config_zloss = dataclasses.replace(
    llama_1_4b_train_config,
    zloss_weight=1e-4,
)


llama_1_4b_train_config_wsd_zloss = dataclasses.replace(
    llama_1_4b_train_config_wsd,
    zloss_weight=1e-4,
)

nemotron_cc_model_wsd = default_train(
    name="nemotron_cc-1.4b-wsd",
    tokenized=nemotron_cc_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config_wsd,
)

nemotron_cc_model_wsd_zloss = default_train(
    name="nemotron_cc-1.4b-wsd-zloss",
    tokenized=nemotron_cc_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config_wsd_zloss,
)

nemotron_cc_model_zloss = default_train(
    name="nemotron_cc-1.4b-zloss",
    tokenized=nemotron_cc_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config_zloss,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            nemotron_cc_model,
            nemotron_cc_model_wsd_zloss,
            nemotron_cc_model_zloss,
            nemotron_cc_model_wsd,
        ],
        description="Train 1.4B models on nemotron-cc using WSD and zloss.",
    )
