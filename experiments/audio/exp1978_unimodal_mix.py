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

"""
Experiment: (Speech, Text) Interleaved and Text-only Data Mix Ablation

1. Total 100B tokens of (Speech, Text) Interleaved and Text-only Data
2. Model size: 150M (Qwen3-150M)
3. Use 0%, 10%, 25%, 50% of the text-only data
"""

import dataclasses
import haliax as hax
from math import ceil

from experiments.qwen3 import qwen3_150m
from experiments.audio.tokenize_emilia import emilia_tokenized_steps
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import SimpleTrainConfig, default_train
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.resources import TpuPodConfig
from levanter.optim import CautiousConfig

SEQ_LEN = 4096
BASE_BATCH_SIZE = 256
BATCH_SIZE = 512
BASE_LEARNING_RATE = 3e-3
LEARNING_RATE = 0.003 * (BATCH_SIZE / BASE_BATCH_SIZE) ** 0.5

audio_unimodal_mix = dataclasses.replace(
    qwen3_150m, tie_word_embeddings=False, gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload")
)

NUM_TRAIN_TOKENS = int(100e9)
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

# (speech,text) interleaved: Emilia-En (37B), Emilia-YODAS-En (73B)
emilia_tokenized = emilia_tokenized_steps()
# text-only: 3.8 trillion tokens https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
dclm_tokenized = dclm_components_llama3["dclm_baseline"]

_EMILIA_WEIGHTS = {
    "Emilia/EN": 0.33669,
    "Emilia-YODAS/EN": 0.66331,
}


def data_mix_config(dclm_weight: float):
    """Create a data mix configuration for the experiment."""
    return lm_mixture_data_config(
        components={
            "Emilia/EN": emilia_tokenized["Emilia/EN"],
            "Emilia-YODAS/EN": emilia_tokenized["Emilia-YODAS/EN"],
            "dclm_baseline": dclm_tokenized,
        },
        weights={
            "Emilia/EN": _EMILIA_WEIGHTS["Emilia/EN"] * (1.00 - dclm_weight),
            "Emilia-YODAS/EN": _EMILIA_WEIGHTS["Emilia-YODAS/EN"] * (1.00 - dclm_weight),
            "dclm_baseline": dclm_weight,
        },
        permutation_type="linear",
    )


data_mixes = {
    "0-100": data_mix_config(0.000),
    "10-90": data_mix_config(0.100),
    "25-75": data_mix_config(0.250),
    "50-50": data_mix_config(0.500),
}


def run_data_mix_experiment():
    results = []
    for name, data_mix in data_mixes.items():
        result = default_train(
            name=f"exp1978_unimodal_mix_{name}",
            tokenized=data_mix,
            model_config=audio_unimodal_mix,
            train_config=training_config,
            eval_harness_tasks=[],
            tags=["audio"],
        )
        results.append(result)
    return results


if __name__ == "__main__":
    executor_main(
        steps=run_data_mix_experiment(),
        description="Experiment: (Speech, Text) Interleaved and Text-only Data Mix Ablation",
    )
