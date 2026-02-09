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

"""Fine-tuning trained model for instruction following using custom instruction datasets."""

import dataclasses
import haliax as hax
from math import ceil

from experiments.qwen3 import qwen3_0_6b_hd128, qwen3_0_6b
from experiments.audio.models import soda_600m_base, blueberry_600m, soda_600m_warmstart, qwen3x_0_6b_base
from experiments.audio.tokenize_sft_cvss import tokenize_instruction_dataset
from experiments.defaults import SimpleTrainConfig, default_train
from marin.processing.tokenize.data_configs import lm_data_config
from marin.execution.executor import executor_main
from levanter.optim import CautiousConfig
from fray.cluster import ResourceConfig

SEQ_LEN = 2048
BATCH_SIZE = 64

NUM_TRAIN_TOKENS = int(750e6)
STEP_PER_EPOCH = ceil(NUM_TRAIN_TOKENS / (BATCH_SIZE * SEQ_LEN))  # 5340 for 2048*64
NUM_EPOCHS = 5
NUM_TRAIN_STEPS = STEP_PER_EPOCH * NUM_EPOCHS
LEARNING_RATE = 2e-5

# Tokenizers for instruction datasets
MARIN_TOKENIZER = "potsawee/marin-mimi-bpe-8cb-16k-tokenizer"
QWEN3X_TOKENIZER = "potsawee/qwen3x-mimi-bpe-8cb-16k-tokenizer"  # Qwen3x models use custom audio tokenizer

# Create tokenized instruction datasets - Marin tokenizer
cvss_method1_marin_tokenized = tokenize_instruction_dataset(MARIN_TOKENIZER, "rma9248/cvss_method1")
cvss_method1_marin_data_config = lm_data_config(training_set=cvss_method1_marin_tokenized)

cvss_method2_marin_tokenized = tokenize_instruction_dataset(MARIN_TOKENIZER, "rma9248/cvss_method2")
cvss_method2_marin_data_config = lm_data_config(training_set=cvss_method2_marin_tokenized)

# Create tokenized instruction datasets - Qwen3x tokenizer
cvss_method1_qwen3x_tokenized = tokenize_instruction_dataset(QWEN3X_TOKENIZER, "rma9248/cvss_method1")
cvss_method1_qwen3x_data_config = lm_data_config(training_set=cvss_method1_qwen3x_tokenized)

cvss_method2_qwen3x_tokenized = tokenize_instruction_dataset(QWEN3X_TOKENIZER, "rma9248/cvss_method2")
cvss_method2_qwen3x_data_config = lm_data_config(training_set=cvss_method2_qwen3x_tokenized)


def finetune_soda_600m_model(dataset_config, dataset_name: str, tpu_type: str = "v5p-8"):
    model_config_600m = dataclasses.replace(
        qwen3_0_6b_hd128,
        tie_word_embeddings=False,
        gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload"),
    )

    optim_config = CautiousConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        min_lr_ratio=0.1,  # Higher min_lr_ratio for SFT
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
        name=f"exp1699_sft_soda_600m_{dataset_name}",
        tokenized=dataset_config,
        model_config=model_config_600m,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["audio-sft", "instruction-following"],
    )


def finetune_soda_600m_warmstart_model(dataset_config, dataset_name: str, tpu_type: str = "v5p-8"):
    model_config_600m = dataclasses.replace(
        qwen3_0_6b_hd128,
        tie_word_embeddings=False,
        gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload"),
    )

    optim_config = CautiousConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        min_lr_ratio=0.1,
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
        name=f"exp1699_sft_soda_600m_warmstart_{dataset_name}",
        tokenized=dataset_config,
        model_config=model_config_600m,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["audio-sft", "instruction-following"],
    )


def finetune_qwen3x_0_6b_base_model(dataset_config, dataset_name: str, tpu_type: str = "v5p-8"):
    model_config_600m = dataclasses.replace(
        qwen3_0_6b_hd128,
        tie_word_embeddings=False,
        gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload"),
    )

    optim_config = CautiousConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        min_lr_ratio=0.1,
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
        name=f"exp1699_sft_qwen3x_0_6b_base_{dataset_name}",
        tokenized=dataset_config,
        model_config=model_config_600m,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["audio-sft", "instruction-following"],
    )


def finetune_blueberry_600m_model(dataset_config, dataset_name: str, tpu_type: str = "v5p-8"):
    model_config_600m = dataclasses.replace(
        qwen3_0_6b,  # Blueberry uses regular qwen3_0_6b, not hd128
        tie_word_embeddings=False,
        gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload"),
    )

    optim_config = CautiousConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        min_lr_ratio=0.1,
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
        name=f"exp1699_sft_blueberry_600m_{dataset_name}",
        tokenized=dataset_config,
        model_config=model_config_600m,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["audio-sft", "instruction-following"],
    )


def finetune_scratch_600m_model(dataset_config, dataset_name: str, tpu_type: str = "v5p-8"):
    model_config_600m = dataclasses.replace(
        qwen3_0_6b_hd128,
        tie_word_embeddings=False,
        gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload"),
    )

    optim_config = CautiousConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        min_lr_ratio=0.1,
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
        # No initialize_from_hf for scratch training
    )

    return default_train(
        name=f"exp1699_sft_scratch_600m_{dataset_name}",
        tokenized=dataset_config,
        model_config=model_config_600m,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["audio-sft", "instruction-following"],
    )


if __name__ == "__main__":
    executor_main(
        steps=[
            # CVSS Method1 experiments - Marin tokenizer models
            finetune_soda_600m_model(cvss_method1_marin_data_config, "cvss_method1", tpu_type="v5p-8"),
            finetune_blueberry_600m_model(cvss_method1_marin_data_config, "cvss_method1", tpu_type="v5p-8"),
            # finetune_soda_600m_warmstart_model(cvss_method1_marin_data_config, "cvss_method1", tpu_type="v5p-8"),
            finetune_scratch_600m_model(cvss_method1_marin_data_config, "cvss_method1", tpu_type="v5p-8"),
            # CVSS Method1 experiments - Qwen3x tokenizer models
            finetune_qwen3x_0_6b_base_model(cvss_method1_qwen3x_data_config, "cvss_method1", tpu_type="v5p-8"),
            # CVSS Method2 experiments - Marin tokenizer models
            # finetune_soda_600m_model(cvss_method2_marin_data_config, "cvss_method2", tpu_type="v5p-8"),
            # finetune_blueberry_600m_model(cvss_method2_marin_data_config, "cvss_method2", tpu_type="v5p-8"),
            # finetune_soda_600m_warmstart_model(cvss_method2_marin_data_config, "cvss_method2", tpu_type="v5p-8"),
            # finetune_scratch_600m_model(cvss_method2_marin_data_config, "cvss_method2", tpu_type="v5p-8"),
            # CVSS Method2 experiments - Qwen3x tokenizer models
            # finetune_qwen3x_0_6b_base_model(cvss_method2_qwen3x_data_config, "cvss_method2", tpu_type="v5p-8"),
        ],
        description="Fine-tune audio models on custom instruction datasets (CVSS method1 & method2).",
    )
