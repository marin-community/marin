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

"""Training config for Data Mix at 150M model size"""

import dataclasses
import haliax as hax
from math import ceil

from levanter.data.text import LMMixtureDatasetConfig
from levanter.optim import CautiousConfig

from experiments.audio.qwen3 import qwen3_150m
from experiments.audio.data_mixes import (
    mix2_v1_english_mixture_config,
    mix3_v1_english_mixture_config,
    mix3_v2_english_mixture_config,
    mix3_v3_english_mixture_config,
)
from experiments.audio.data_mixes import (
    mix3_v4_english_mixture_config,
    mix2_v2_english_mixture_config,
    mix2_v3_english_mixture_config,
)
from experiments.audio.tokenize_yodas import yodas2_english_data_config
from experiments.audio.tokenize_emilia import emilia_english_mixture_config
from experiments.audio.tokenize_mls_en import mls_en_data_config
from experiments.defaults import SimpleTrainConfig, default_train
from marin.processing.tokenize.data_configs import lm_data_config
from experiments.audio.tokenize_cooldown import tokenize_peoples_speech_steps
from experiments.audio.tokenize_cooldown import tokenize_common_voice_17_steps
from experiments.audio.tokenize_cooldown import tokenize_librispeech_steps
from experiments.audio.tokenize_cooldown import tokenize_libritts_steps

from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig

SEQ_LEN = 4096
BASE_LEARNING_RATE = 3e-3
BASE_BATCH_SIZE = 256
BATCH_SIZE = 128
BASE_WIDTH = 1024
WIDTH = 512
LEARNING_RATE = BASE_LEARNING_RATE * (BATCH_SIZE / BASE_BATCH_SIZE) ** 0.5 * (WIDTH / BASE_WIDTH) ** 0.5
yodas_qwen_150m = dataclasses.replace(
    qwen3_150m, gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload")
)

# NUM_TRAIN_TOKENS = int(10e9)
NUM_TRAIN_TOKENS = int(3e9)
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
    resources=ResourceConfig.with_tpu("v5p-8"),
    train_batch_size=BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    z_loss_weight=1e-4,
    optimizer_config=optim_config,
)


def generate_train_config(
    name: str,
    data_mix_config: LMMixtureDatasetConfig,
):
    return default_train(
        name=name,
        tokenized=data_mix_config,
        model_config=yodas_qwen_150m,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["data_mix"],
    )


if __name__ == "__main__":
    mix2_v1_model = generate_train_config(
        name="exp1699_data_mix_150m_mix2_v1",
        data_mix_config=mix2_v1_english_mixture_config(),
    )
    mix3_v1_model = generate_train_config(
        name="exp1699_data_mix_150m_mix3_v1",
        data_mix_config=mix3_v1_english_mixture_config(),
    )
    yodas_model = generate_train_config(
        name="exp1699_data_mix_150m_yodas",
        data_mix_config=yodas2_english_data_config(),
    )
    emilia_model = generate_train_config(
        name="exp1699_data_mix_150m_emilia",
        data_mix_config=emilia_english_mixture_config(),
    )
    mls_en_model = generate_train_config(
        name="exp1699_data_mix_150m_mls_en",
        data_mix_config=mls_en_data_config(),
    )
    mix3_v2_model = generate_train_config(
        name="exp1699_data_mix_150m_mix3_v2",
        data_mix_config=mix3_v2_english_mixture_config(),
    )
    mix3_v3_model = generate_train_config(
        name="exp1699_data_mix_150m_mix3_v3",
        data_mix_config=mix3_v3_english_mixture_config(),
    )
    mix3_v4_model = generate_train_config(
        name="exp1699_data_mix_150m_mix3_v4",
        data_mix_config=mix3_v4_english_mixture_config(),
    )
    mix2_v2_model = generate_train_config(
        name="exp1699_data_mix_150m_mix2_v2",
        data_mix_config=mix2_v2_english_mixture_config(),
    )
    mix2_v3_model = generate_train_config(
        name="exp1699_data_mix_150m_mix2_v3",
        data_mix_config=mix2_v3_english_mixture_config(),
    )
    steps = []
    steps += [mix2_v1_model, mix3_v1_model, yodas_model, emilia_model, mls_en_model, mix3_v2_model]
    steps += [mix3_v3_model]
    steps += [mix3_v4_model, mix2_v2_model, mix2_v3_model]

    yodas_model = generate_train_config(
        name="exp1699_data_mix_150m_3B_yodas",
        data_mix_config=yodas2_english_data_config(),
    )
    emilia_model = generate_train_config(
        name="exp1699_data_mix_150m_3B_emilia",
        data_mix_config=emilia_english_mixture_config(),
    )
    mls_en_model = generate_train_config(
        name="exp1699_data_mix_150m_3B_mls_en",
        data_mix_config=mls_en_data_config(),
    )
    # cooldown mix
    peoples_speech_tokenized = tokenize_peoples_speech_steps()["peoples-speech-clean"]
    peoples_speech_data_config = lm_data_config(training_set=peoples_speech_tokenized)
    common_voice_17_tokenized = tokenize_common_voice_17_steps()["commonvoice17-en"]
    common_voice_17_data_config = lm_data_config(training_set=common_voice_17_tokenized)
    librispeech_tokenized = tokenize_librispeech_steps()["librispeech-train"]
    librispeech_data_config = lm_data_config(training_set=librispeech_tokenized)
    libritts_tokenized = tokenize_libritts_steps()["libritts-train"]
    libritts_data_config = lm_data_config(training_set=libritts_tokenized)

    peoples_speech_model = generate_train_config(
        name="exp1699_data_mix_150m_3B_peoples_speech",
        data_mix_config=peoples_speech_data_config,
    )
    common_voice_17_model = generate_train_config(
        name="exp1699_data_mix_150m_3B_commonvoice17",
        data_mix_config=common_voice_17_data_config,
    )
    librispeech_model = generate_train_config(
        name="exp1699_data_mix_150m_3B_librispeech",
        data_mix_config=librispeech_data_config,
    )
    libritts_model = generate_train_config(
        name="exp1699_data_mix_150m_3B_libritts",
        data_mix_config=libritts_data_config,
    )
    steps += [
        yodas_model,
        emilia_model,
        mls_en_model,
        peoples_speech_model,
        common_voice_17_model,
        librispeech_model,
        libritts_model,
    ]

    executor_main(
        steps=steps,
        description="Train the 150M model with different data mix configurations.",
    )
