"""
Experiment 934: See if we should make zloss be default on
"""

import dataclasses

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3
from experiments.defaults import default_train
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
    z_loss_weight=1e-4,
)


llama_1_4b_train_config_wsd_zloss = dataclasses.replace(
    llama_1_4b_train_config_wsd,
    z_loss_weight=1e-4,
)


dclm_mix_model = default_train(
    name="dclm_mix-cosine-1.4b",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

dclm_mix_model_wsd = default_train(
    name="dclm_mix-1.4b-wsd",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config_wsd,
)

dclm_mix_model_wsd_zloss = default_train(
    name="dclm_mix-1.4b-wsd-zloss",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config_wsd_zloss,
)

dclm_mix_model_zloss = default_train(
    name="dclm_mix-1.4b-zloss",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config_zloss,
)


if __name__ == "__main__":
    executor_main(
        steps=[
            dclm_mix_model,
            dclm_mix_model_wsd_zloss,
            dclm_mix_model_zloss,
            dclm_mix_model_wsd,
        ],
        description="Train 1.4B models on dclm using WSD and zloss.",
    )
