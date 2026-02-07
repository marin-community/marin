# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training config for the Marin-Audio (Qwen3x) warm-start models."""

import dataclasses
import haliax as hax
from math import ceil

from experiments.qwen3 import qwen3_1_7b, qwen3_0_6b_hd128
from experiments.audio.models import qwen3x_0_6b_base, qwen3x_1_7b_base
from experiments.audio.tokenize_qwen3x_audio import (
    tokenize_yodas_steps,
    tokenize_emilia_steps,
    tokenize_nemotron_hq_actual_step,
)
from experiments.defaults import SimpleTrainConfig, default_train
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig
from levanter.optim import CautiousConfig

SEQ_LEN = 4096
BASE_BATCH_SIZE = 256
BATCH_SIZE = 512
BASE_WIDTH = 1024
BASE_LEARNING_RATE = 0.003

NUM_TRAIN_TOKENS = int(500e9)
NUM_TRAIN_STEPS = ceil(NUM_TRAIN_TOKENS / (BATCH_SIZE * SEQ_LEN))


# Data mix configuration for the Marin Yodas2 audio model
# Speech-Text: Yodas2-En (131B) + Emilia-Yodas2-En (73B) + Emilia-En (37B)
# Text-only: Nemotron-HighActual at 5% ratio
yodas2_en_tokenized = tokenize_yodas_steps()["yodas2/en"]
emilia_yodas_en_tokenized = tokenize_emilia_steps()["Emilia-YODAS/EN"]
emilia_en_tokenized = tokenize_emilia_steps()["Emilia/EN"]
nemotron_tokenized = tokenize_nemotron_hq_actual_step()

text_ratio = 0.05
speech_text_ratio = 1.00 - text_ratio

data_mix_config = lm_mixture_data_config(
    components={
        "yodas2/en": yodas2_en_tokenized,
        "Emilia-YODAS/EN": emilia_yodas_en_tokenized,
        "Emilia/EN": emilia_en_tokenized,
        "nemotron_cc/hq_actual": nemotron_tokenized,
    },
    weights={
        "yodas2/en": speech_text_ratio * (131 / (131 + 73 + 37)),
        "Emilia-YODAS/EN": speech_text_ratio * (73 / (131 + 73 + 37)),
        "Emilia/EN": speech_text_ratio * (37 / (131 + 73 + 37)),
        "nemotron_cc/hq_actual": text_ratio,
    },
    permutation_type="feistel",
)


def train_1_7b_model(tpu_type: str = "v5p-128"):
    width = 2048
    learning_rate = BASE_LEARNING_RATE * (BATCH_SIZE / BASE_BATCH_SIZE) ** 0.5 * (BASE_WIDTH / width) ** 0.5
    yodas_qwen_1_7b = dataclasses.replace(
        qwen3_1_7b, tie_word_embeddings=False, gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload")
    )
    optim_config = CautiousConfig(
        learning_rate=learning_rate,
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
        resources=ResourceConfig.with_tpu(tpu_type),
        train_batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=learning_rate,
        z_loss_weight=1e-4,
        optimizer_config=optim_config,
        initialize_from_hf=qwen3x_1_7b_base,
    )
    return default_train(
        name="exp1699_marin_audio_ws_1_7b",
        tokenized=data_mix_config,
        model_config=yodas_qwen_1_7b,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["audio"],
    )


def train_600m_model(tpu_type: str = "v5p-64"):
    width = 1024
    learning_rate = BASE_LEARNING_RATE * (BATCH_SIZE / BASE_BATCH_SIZE) ** 0.5 * (BASE_WIDTH / width) ** 0.5
    yodas_qwen_600m = dataclasses.replace(
        qwen3_0_6b_hd128,
        tie_word_embeddings=False,
        gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload"),
    )
    optim_config = CautiousConfig(
        learning_rate=learning_rate,
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
        resources=ResourceConfig.with_tpu(tpu_type),
        train_batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=learning_rate,
        z_loss_weight=1e-4,
        optimizer_config=optim_config,
        initialize_from_hf=qwen3x_0_6b_base,
    )
    return default_train(
        name="exp1699_marin_audio_ws_600m",
        tokenized=data_mix_config,
        model_config=yodas_qwen_600m,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["audio"],
    )


if __name__ == "__main__":
    marin_audio_1_7b_model = train_1_7b_model(tpu_type="v5p-128")
    marin_audio_600m_model = train_600m_model(tpu_type="v5p-64")

    executor_main(
        steps=[marin_audio_1_7b_model, marin_audio_600m_model],
        description="Train the Marin-Audio (Qwen3x) warm-start models.",
    )
