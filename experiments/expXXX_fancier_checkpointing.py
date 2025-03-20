"""
Test different checkpointing strategies for the 1.4b models.
"""
import dataclasses
import logging
import math

from haliax.nn.scan import StackedCheckpointPolicy
from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3
from experiments.defaults import default_train
from experiments.llama import llama_1_4b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main, versioned

logger = logging.getLogger("ray")

BEST_LR = 3e-3 / 4
WD = 0.1
TPU_TYPE = "v4-128"
TOKEN_TARGETS = 50_000_000
BATCH_SIZE = 1024
SEQ_LEN = 4096


def step_target(token_target, batch_size, seq_len):
    actual_step_count = math.ceil(token_target / (batch_size * seq_len))
    nice_round_step_count = math.ceil(actual_step_count / 1000) * 1000
    return nice_round_step_count


num_train_steps = step_target(TOKEN_TARGETS, BATCH_SIZE, SEQ_LEN)

baseline_train_config = SimpleTrainConfig(
    tpu_type=versioned(TPU_TYPE),
    train_batch_size=versioned(BATCH_SIZE),
    num_train_steps=num_train_steps,
    learning_rate=BEST_LR,
    weight_decay=WD,
)


baseline_1b = default_train(
    name="fancy_ckpt-v4-1b-baseline",
    train_config=baseline_train_config,
    model_config=llama_1_4b,
    tokenized=dclm_mixture_config_llama3,
    tags=("llama", "1.4b", "fancy_ckpt", "dclm"),
    eval_harness_tasks=[],
)

offload_policy = StackedCheckpointPolicy.from_bool_or_str("offload")
full_policy = StackedCheckpointPolicy.from_bool_or_str("full")

llama_offload = dataclasses.replace(llama_1_4b, gradient_checkpointing=offload_policy)

llama_full = dataclasses.replace(llama_1_4b, gradient_checkpointing=full_policy)

offload_1b = default_train(
    name="fancy_ckpt-v4-1b-offload",
    train_config=baseline_train_config,
    model_config=llama_offload,
    tokenized=dclm_mixture_config_llama3,
    tags=("llama", "1.4b", "fancy_ckpt", "dclm"),
    eval_harness_tasks=[],
)

# full_1b = default_train(
#     name="fancy_ckpt-v4-1b-full",
#     train_config=baseline_train_config,
#     model_config=llama_full,
#     tokenized=dclm_mixture_config_llama3,
#     tags=("llama", "1.4b", "fancy_ckpt", "dclm"),
#     eval_harness_tasks=[],
# )


if __name__ == "__main__":
    executor_main(
        steps=[baseline_1b, offload_1b], #, full_1b],
        description=""" Test different checkpointing strategies for the 1.4b models.""",
    )
