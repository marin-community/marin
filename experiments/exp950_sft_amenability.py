"""
Experiment 934: See if we should make zloss be default on
"""

import dataclasses

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3
from experiments.defaults import default_train
from experiments.exp606_sft import tulu_sft_config
from experiments.llama import llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import executor_main, output_path_of

llama_1_4b_wsd_high_lr_train_config = dataclasses.replace(
    llama_1_4b_train_config,
    num_train_steps=238418,  # 4096 * 1024 * 238418 = 1T tokens
    weight_decay=0.05,
    decay=0.2,
    learning_rate=1e-3,
    lr_schedule="linear",
    ema_beta=0.995,
    z_loss_weight=1e-4,
)


llama_1_4b_cos_high_lr_train_config = dataclasses.replace(
    llama_1_4b_wsd_high_lr_train_config,
    lr_schedule="cosine",
)


llama_1_4b_cos_olmo_lr_train_config = dataclasses.replace(llama_1_4b_cos_high_lr_train_config, learning_rate=3e-4)


dclm_mix_model_wsd = default_train(
    name="lr_tests_wsd",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_1_4b,
    train_config=llama_1_4b_wsd_high_lr_train_config,
)

dclm_mix_model_cos_high = default_train(
    name="lr_tests_cos_high",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_1_4b,
    train_config=llama_1_4b_cos_high_lr_train_config,
)

dclm_mix_model_cos_low = default_train(
    name="lr_tests_cos_low",
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_1_4b,
    train_config=llama_1_4b_cos_olmo_lr_train_config,
)

sft_model_wsd = dataclasses.replace(
    tulu_sft_config,
    model_name_or_path=output_path_of(dclm_mix_model_wsd, "hf/238417/"),
)
sft_model_cos_high = dataclasses.replace(
    tulu_sft_config,
    model_name_or_path=output_path_of(dclm_mix_model_cos_high, "hf/238417/"),
)
sft_model_cos_low = dataclasses.replace(
    tulu_sft_config,
    model_name_or_path=output_path_of(dclm_mix_model_cos_low, "hf/238417/"),
)

if __name__ == "__main__":
    executor_main(
        steps=[
            dclm_mix_model_wsd,
            dclm_mix_model_cos_high,
            dclm_mix_model_cos_low,
            sft_model_wsd,
            sft_model_cos_high,
            sft_model_cos_low,
        ],
        description="Train 1.4B models on dclm using varying learning rates, then SFT the resulting models.",
    )
