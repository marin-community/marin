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

"""Training config for the Marin-Audio-135 annealing experiment."""

import dataclasses
import haliax as hax

from experiments.anneal_config import AnnealConfig
from experiments.audio.qwen3 import qwen3_135m
from experiments.audio.audio_defaults import default_audio_anneal
from experiments.audio.exp1699_marin_audio_all import train_135m_model
from experiments.audio.tokenize_yodas import yodas2_tokenized_steps
from experiments.audio.tokenize_emilia import tokenize_emilia_fix_steps
from experiments.audio.tokenize_mls_en import tokenize_mls_en_steps
from experiments.audio.tokenize_nemotron import tokenize_nemotron_hq_actual_step
from marin.processing.tokenize.data_configs import lm_mixture_data_config

from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig

SEQ_LEN = 4096
BASE_BATCH_SIZE = 256
BATCH_SIZE = 512
BASE_LEARNING_RATE = 3e-3
BASE_WIDTH = 1024
WIDTH = 1024  # it is actually 512, but I don't want LR to be too high
LEARNING_RATE = BASE_LEARNING_RATE * (BATCH_SIZE / BASE_BATCH_SIZE) ** 0.5 * (BASE_WIDTH / WIDTH) ** 0.5
NUM_ANNEAL_STEPS = 50000
NUM_ANNEAL_TRAINING_TOKENS = int(NUM_ANNEAL_STEPS * BATCH_SIZE * SEQ_LEN)

checkpoint_step = 190000

marin_audio_135m_model = train_135m_model(tpu_type="v5p-32")
marin_audio_135m_model_correct = marin_audio_135m_model.with_output_path("checkpoints/exp1699_marin_audio_135m-b0c7b0")
train_step = marin_audio_135m_model_correct.cd(f"checkpoints/step-{checkpoint_step}")

qwen3_135m_config = dataclasses.replace(
    qwen3_135m, tie_word_embeddings=False, gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload")
)

# Data mix configuration for the Marin Yodas2 audio model
# Speech-Text: Yodas2-En (131B) + Emilia-Yodas2-En (73B) + Emilia-En (37B)
# Text-only: Nemotron-HighActual at 5% ratio
yodas2_en_tokenized = yodas2_tokenized_steps()["yodas2/en"]
emilia_yodas_en_tokenized = tokenize_emilia_fix_steps()["Emilia-YODAS/EN"]
emilia_en_tokenized = tokenize_emilia_fix_steps()["Emilia/EN"]
mls_en_tokenized = tokenize_mls_en_steps()["mls-en"]
nemotron_tokenized = tokenize_nemotron_hq_actual_step()

text_ratio = 0.05
speech_text_ratio = 1.00 - text_ratio

data_mix3_v3_config = lm_mixture_data_config(
    components={
        "mls-en": mls_en_tokenized,
        "emilia-en": emilia_en_tokenized,
        "emilia-yodas-en": emilia_yodas_en_tokenized,
        "yodas2-en": yodas2_en_tokenized,
        "nemotron_cc/hq_actual": nemotron_tokenized,
    },
    weights={
        "mls-en": 0.33333 * speech_text_ratio,
        "emilia-en": 0.33333 * speech_text_ratio,
        "emilia-yodas-en": 0.33333 * (73 / (73 + 131)) * speech_text_ratio,
        "yodas2-en": 0.33333 * (131 / (73 + 131)) * speech_text_ratio,
        "nemotron_cc/hq_actual": text_ratio,
    },
    permutation_type="feistel",
)

data_mix2_v2_config = lm_mixture_data_config(
    components={
        "mls-en": mls_en_tokenized,
        "emilia-en": emilia_en_tokenized,
        "emilia-yodas-en": emilia_yodas_en_tokenized,
        "nemotron_cc/hq_actual": nemotron_tokenized,
    },
    weights={
        "mls-en": 0.33333 * speech_text_ratio,
        "emilia-en": 0.33333 * speech_text_ratio,
        "emilia-yodas-en": 0.33333 * speech_text_ratio,
        "nemotron_cc/hq_actual": text_ratio,
    },
    permutation_type="feistel",
)

data_mix3_v5_yodas05_config = lm_mixture_data_config(
    components={
        "mls-en": mls_en_tokenized,
        "emilia-en": emilia_en_tokenized,
        "emilia-yodas-en": emilia_yodas_en_tokenized,
        "yodas2-en": yodas2_en_tokenized,
        "nemotron_cc/hq_actual": nemotron_tokenized,
    },
    weights={
        "mls-en": 0.33333 * speech_text_ratio,
        "emilia-en": 0.33333 * speech_text_ratio,
        "emilia-yodas-en": 0.119 * speech_text_ratio,
        "yodas2-en": 0.214 * 0.05 * speech_text_ratio,
        "nemotron_cc/hq_actual": text_ratio,
    },
    permutation_type="feistel",
)


# Mix3-v3 Anneal
annealing_mix3_v3_en_config = AnnealConfig(
    initialize_from_checkpoint_path=train_step,
    dataset_config=data_mix3_v3_config,
    learning_rate=LEARNING_RATE,
    weight_decay=0.033,
    min_lr_ratio=0.0,
    lr_schedule="linear",
    train_batch_size=BATCH_SIZE,
    num_anneal_training_tokens=NUM_ANNEAL_TRAINING_TOKENS,
    resources=ResourceConfig.with_tpu("v5p-32"),
)
annealed_model_mix3_v3_en = default_audio_anneal(
    name="exp1699_marin_audio_135m_anneal_nemo_mix3_v3",
    model_config=qwen3_135m_config,
    anneal_config=annealing_mix3_v3_en_config,
)

# Mix2-v2 Anneal
annealing_mix2_v2_en_config = AnnealConfig(
    initialize_from_checkpoint_path=train_step,
    dataset_config=data_mix2_v2_config,
    learning_rate=LEARNING_RATE,
    weight_decay=0.033,
    min_lr_ratio=0.0,
    lr_schedule="linear",
    train_batch_size=BATCH_SIZE,
    num_anneal_training_tokens=NUM_ANNEAL_TRAINING_TOKENS,
    resources=ResourceConfig.with_tpu("v5p-32"),
)
annealed_model_mix2_v2_en = default_audio_anneal(
    name="exp1699_marin_audio_135m_anneal_nemo_mix2_v2",
    model_config=qwen3_135m_config,
    anneal_config=annealing_mix2_v2_en_config,
)

# Mix3-v5-Yodas05 Anneal
annealing_mix3_v5_yodas05_en_config = AnnealConfig(
    initialize_from_checkpoint_path=train_step,
    dataset_config=data_mix3_v5_yodas05_config,
    learning_rate=LEARNING_RATE,
    weight_decay=0.033,
    min_lr_ratio=0.0,
    lr_schedule="linear",
    train_batch_size=BATCH_SIZE,
    num_anneal_training_tokens=NUM_ANNEAL_TRAINING_TOKENS,
    resources=ResourceConfig.with_tpu("v5p-32"),
)
annealed_model_mix3_v5_yodas05_en = default_audio_anneal(
    name="exp1699_marin_audio_135m_anneal_nemo_mix3_v5_yodas05",
    model_config=qwen3_135m_config,
    anneal_config=annealing_mix3_v5_yodas05_en_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[annealed_model_mix3_v3_en, annealed_model_mix2_v2_en, annealed_model_mix3_v5_yodas05_en],
        description="Cooldown the 135M model with Mix3-v3, Mix2-v2, and Mix3-v5-Yodas05 En-only with Nemotron.",
    )
