# https://github.com/stanford-crfm/marin/issues/621
# Sweep to determine optimal training configs for small models
import dataclasses
import logging
import math

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3
from experiments.defaults import default_train
from experiments.llama import llama_1_4b, llama_8b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main, versioned

logger = logging.getLogger("ray")

BEST_LR = 3e-3 / 4
WD = 0.1
TPU_TYPE = "v5litepod-256"
TOKEN_TARGETS = 40_000_000_000
BATCH_SIZE = 1024
SEQ_LEN = 4096


def step_target(token_target, batch_size, seq_len):
    actual_step_count = math.ceil(token_target / (batch_size * seq_len))
    nice_round_step_count = math.ceil(actual_step_count / 1000) * 1000
    return nice_round_step_count


num_train_steps = step_target(TOKEN_TARGETS, BATCH_SIZE, SEQ_LEN)

baseline_train_config = SimpleTrainConfig(
    tpu_type=versioned(TPU_TYPE),
    train_batch_size=BATCH_SIZE,
    num_train_steps=num_train_steps,
    learning_rate=BEST_LR,
    weight_decay=WD,
)
int8_train_config = dataclasses.replace(baseline_train_config, int8=True)

baseline_1b_step = default_train(
    name="exp620-v5e-1.4b-baseline-profile",
    train_config=baseline_train_config,
    model_config=llama_1_4b,
    tokenized=dclm_mixture_config_llama3,
    tags=("llama", "1.4b", "620_int8", "dclm"),
)

int8_1b_step = default_train(
    name="exp620-v5e-1.4b-int8-maxtext",
    train_config=int8_train_config,
    model_config=llama_1_4b,
    tokenized=dclm_mixture_config_llama3,
    tags=("llama", "1.4b", "620_int8", "dclm"),
)

baseline_8b_step = default_train(
    name="exp620-v5e-8b-baseline",
    train_config=baseline_train_config,
    model_config=llama_8b,
    tokenized=dclm_mixture_config_llama3,
    tags=("llama", "8b", "620_int8", "dclm"),
)

int8_8b_step = default_train(
    name="exp620-v5e-8b-int8",
    train_config=int8_train_config,
    model_config=llama_8b,
    tokenized=dclm_mixture_config_llama3,
    tags=("llama", "8b", "620_int8", "dclm"),
)


if __name__ == "__main__":
    executor_main([baseline_1b_step, int8_1b_step, baseline_8b_step, int8_8b_step])
