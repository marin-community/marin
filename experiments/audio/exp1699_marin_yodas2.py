"""Training config for the Marin Yodas2 audio run targeting 500B tokens."""

import dataclasses
from math import ceil

from experiments.qwen3 import qwen3_0_6b
from experiments.audio.tokenize_yodas import yodas2_mixture_config
from experiments.defaults import SimpleTrainConfig, default_train
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from levanter.optim import CautiousConfig

SEQ_LEN = 4096
BASE_BATCH_SIZE = 256
BATCH_SIZE = 2048  # leverage v5p-64 capacity
BASE_LEARNING_RATE = 3e-3
LEARNING_RATE = 0.003

yodas_qwen = dataclasses.replace(qwen3_0_6b, tie_word_embeddings=False)

NUM_TRAIN_TOKENS = int(500e9)
NUM_TRAIN_STEPS = ceil(NUM_TRAIN_TOKENS / (BATCH_SIZE * SEQ_LEN))

optim_config = CautiousConfig(
    learning_rate=LEARNING_RATE,
    weight_decay=0.033,
    min_lr_ratio=0.0,
    warmup=0.1,
    decay=0.2,
    beta1=0.98,
    beta2=0.98,
    epsilon=1e-16,
    max_grad_norm=1,
    lr_schedule="linear",
    adamc_weight_decay=True,
)

training_config = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v5p-64"),
    train_batch_size=BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    z_loss_weight=1e-4,
    optimizer_config=optim_config,
)

yodas_1b_model = default_train(
    name="exp1699_marin_yodas2",
    tokenized=yodas2_mixture_config(),
    model_config=yodas_qwen,
    train_config=training_config,
    tags=["AUDIO", "MARIN_YODAS2"],
)

if __name__ == "__main__":
    executor_main(
        steps=[yodas_1b_model],
        description="Train the Marin Yodas2 audio model on 350B tokens with v5p-64.",
    )
