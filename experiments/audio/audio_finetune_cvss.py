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

"""Fine-tuning trained model for Speech-to-Speech translation."""

import dataclasses
import haliax as hax
from math import ceil

from experiments.qwen3 import qwen3_0_6b_hd128, qwen3_0_6b
from experiments.audio.models import soda_600m_base, blueberry_600m, soda_600m_warmstart, qwen3x_0_6b_base
from experiments.audio.tokenize_finetune import tokenize_cvss_method1_steps as tokenize_cvss_method1_steps_marin
from experiments.audio.tokenize_qwen3x_audio import tokenize_cvss_method1_steps as tokenize_cvss_method1_steps_qwen3x
from experiments.defaults import SimpleTrainConfig, default_train
from marin.processing.tokenize.data_configs import lm_data_config
from marin.execution.executor import executor_main
from levanter.optim import CautiousConfig
from fray.cluster import ResourceConfig

SEQ_LEN = 2048
BATCH_SIZE = 64

NUM_TRAIN_TOKENS = int(700e6)
STEP_PER_EPOCH = ceil(NUM_TRAIN_TOKENS / (BATCH_SIZE * SEQ_LEN))  # 5340 for 2048*64
NUM_EPOCHS = 5
NUM_TRAIN_STEPS = STEP_PER_EPOCH * NUM_EPOCHS
LEARNING_RATE = 2e-5


cvss_method1_marin_tokenized = tokenize_cvss_method1_steps_marin()["cvss-method1"]
cvss_method1_marin_data_config = lm_data_config(training_set=cvss_method1_marin_tokenized)

cvss_method1_qwen3x_tokenized = tokenize_cvss_method1_steps_qwen3x()["cvss-method1"]
cvss_method1_qwen3x_data_config = lm_data_config(training_set=cvss_method1_qwen3x_tokenized)


def finetune_soda_600m_model(tpu_type: str = "v5p-8"):
    model_config_600m = dataclasses.replace(
        qwen3_0_6b_hd128,
        tie_word_embeddings=False,
        gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload"),
    )

    optim_config = CautiousConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        min_lr_ratio=0.0,
        warmup=0.03,
        decay=1.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        max_grad_norm=1,
        lr_schedule="cosine",
        adamc_weight_decay=True,
    )

    training_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type),
        train_batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        z_loss_weight=1e-4,
        optimizer_config=optim_config,
        initialize_from_hf=soda_600m_base,
    )

    return default_train(
        name="exp1699_finetune_soda_600m_cvss_method1",
        tokenized=cvss_method1_marin_data_config,
        model_config=model_config_600m,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["audio-finetune"],
    )


def finetune_soda_600m_warmstart_model(tpu_type: str = "v5p-8"):
    model_config_600m = dataclasses.replace(
        qwen3_0_6b_hd128,
        tie_word_embeddings=False,
        gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload"),
    )

    optim_config = CautiousConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        min_lr_ratio=0.0,
        warmup=0.03,
        decay=1.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        max_grad_norm=1,
        lr_schedule="cosine",
        adamc_weight_decay=True,
    )

    training_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type),
        train_batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        z_loss_weight=1e-4,
        optimizer_config=optim_config,
        initialize_from_hf=soda_600m_warmstart,
    )

    return default_train(
        name="exp1699_finetune_soda_600m_warmstart_cvss_method1",
        tokenized=cvss_method1_qwen3x_data_config,
        model_config=model_config_600m,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["audio-finetune"],
    )


def finetune_qwen3x_0_6b_base_model(tpu_type: str = "v5p-8"):
    model_config_600m = dataclasses.replace(
        qwen3_0_6b_hd128,
        tie_word_embeddings=False,
        gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload"),
    )

    optim_config = CautiousConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        min_lr_ratio=0.0,
        warmup=0.03,
        decay=1.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        max_grad_norm=1,
        lr_schedule="cosine",
        adamc_weight_decay=True,
    )

    training_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type),
        train_batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        z_loss_weight=1e-4,
        optimizer_config=optim_config,
        initialize_from_hf=qwen3x_0_6b_base,
    )

    return default_train(
        name="exp1699_finetune_qwen3x_0_6b_base_cvss_method1",
        tokenized=cvss_method1_qwen3x_data_config,
        model_config=model_config_600m,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["audio-finetune"],
    )


def finetune_blueberry_600m_model(tpu_type: str = "v5p-8"):
    model_config_600m = dataclasses.replace(
        qwen3_0_6b,
        tie_word_embeddings=False,
        gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload"),
    )

    optim_config = CautiousConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        min_lr_ratio=0.0,
        warmup=0.03,
        decay=1.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        max_grad_norm=1,
        lr_schedule="cosine",
        adamc_weight_decay=True,
    )

    training_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type),
        train_batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        z_loss_weight=1e-4,
        optimizer_config=optim_config,
        initialize_from_hf=blueberry_600m,
    )

    return default_train(
        name="exp1699_finetune_blueberry_600m_cvss_method1",
        tokenized=cvss_method1_marin_data_config,
        model_config=model_config_600m,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["audio-finetune"],
    )


def finetune_scratch_600m_model(tpu_type: str = "v5p-8"):
    model_config_600m = dataclasses.replace(
        qwen3_0_6b_hd128,
        tie_word_embeddings=False,
        gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload"),
    )

    optim_config = CautiousConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        min_lr_ratio=0.0,
        warmup=0.03,
        decay=1.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        max_grad_norm=1,
        lr_schedule="cosine",
        adamc_weight_decay=True,
    )

    training_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type),
        train_batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        z_loss_weight=1e-4,
        optimizer_config=optim_config,
    )

    return default_train(
        name="exp1699_finetune_scratch_600m_cvss_method1",
        tokenized=cvss_method1_marin_data_config,
        model_config=model_config_600m,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["audio-finetune"],
    )


if __name__ == "__main__":
    executor_main(
        steps=[
            finetune_soda_600m_model(tpu_type="v5p-8"),
            finetune_blueberry_600m_model(tpu_type="v5p-8"),
            finetune_soda_600m_warmstart_model(tpu_type="v5p-8"),
            finetune_qwen3x_0_6b_base_model(tpu_type="v5p-8"),
            finetune_scratch_600m_model(tpu_type="v5p-8"),
        ],
        description="Finetune SODA models on CVSS-method1.",
    )
