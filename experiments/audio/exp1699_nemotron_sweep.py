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

"""Training config for the Marin-Yodas2 audio run with Qwen3."""

import dataclasses
import haliax as hax
from math import ceil

from experiments.audio.qwen3 import qwen3_30m, qwen3_50m, qwen3_75m, qwen3_150m
from experiments.audio.tokenize_yodas import yodas2_tokenized_steps
from experiments.audio.tokenize_nemotron import tokenize_nemotron_hq_actual_step
from experiments.defaults import SimpleTrainConfig, default_train
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from fray.cluster import ResourceConfig
from levanter.optim import CautiousConfig

MODEL_SIZE_IN_MILLION_PARAMS = 150
NUM_TRAIN_TOKENS = int(10e9)

if MODEL_SIZE_IN_MILLION_PARAMS == 30:
    model_config = qwen3_30m
elif MODEL_SIZE_IN_MILLION_PARAMS == 50:
    model_config = qwen3_50m
elif MODEL_SIZE_IN_MILLION_PARAMS == 75:
    model_config = qwen3_75m
elif MODEL_SIZE_IN_MILLION_PARAMS == 150:
    model_config = qwen3_150m
else:
    raise ValueError(f"Unknown model size: {MODEL_SIZE_IN_MILLION_PARAMS}")


SEQ_LEN = 4096
BASE_BATCH_SIZE = 256
BATCH_SIZE = 128
LEARNING_RATE = 0.003 * (BATCH_SIZE / BASE_BATCH_SIZE) ** 0.5

model_config = dataclasses.replace(model_config, gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload"))

NUM_TRAIN_STEPS = ceil(NUM_TRAIN_TOKENS / (BATCH_SIZE * SEQ_LEN))
_NUM_TRAIN_TOKENS_IN_BILLIONS = int(NUM_TRAIN_TOKENS / 1e9)

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

# (speech,text) interleaved: Yodas2-En (131B)
# use Marin tokenizer (Llama) so that we won't need to tokenize the nemotron data
yodas2_en_tokenized = yodas2_tokenized_steps()["yodas2/en"]

# nemotron (text-only pre-training data) tokenized
# nemotron_tokenized = tokenize_nemotron_steps()
nemotron_tokenized = tokenize_nemotron_hq_actual_step()


def data_mix_config(text_data_weight: float):
    """Create a data mix configuration for the experiment."""
    return lm_mixture_data_config(
        components={
            "yodas2/en": yodas2_en_tokenized,
            # "nemotron_cc/hq_actual": nemotron_tokenized["nemotron_cc/hq_actual"]
            "nemotron_cc/hq_actual": nemotron_tokenized,
        },
        weights={
            "yodas2/en": 1.00 - text_data_weight,
            "nemotron_cc/hq_actual": text_data_weight,
        },
        permutation_type="linear",
    )


data_mixes = {
    "0-100": data_mix_config(0.000),
    "10-90": data_mix_config(0.100),
    "20-80": data_mix_config(0.200),
    "30-70": data_mix_config(0.300),
    "40-60": data_mix_config(0.400),
    "50-50": data_mix_config(0.500),
    "60-40": data_mix_config(0.600),
    "70-30": data_mix_config(0.700),
    "80-20": data_mix_config(0.800),
    "90-10": data_mix_config(0.900),
    "100-0": data_mix_config(1.000),
    "2.5-97.5": data_mix_config(0.025),
    "5-95": data_mix_config(0.05),
    "7.5-92.5": data_mix_config(0.075),
}


def run_data_mix_experiment():
    results = []
    for name, data_mix in data_mixes.items():
        run_name = (
            f"exp1699_nemotron_sweep_{MODEL_SIZE_IN_MILLION_PARAMS}M_tok{_NUM_TRAIN_TOKENS_IN_BILLIONS}B_mix{name}"
        )
        result = default_train(
            name=run_name,
            tokenized=data_mix,
            model_config=model_config,
            train_config=training_config,
            eval_harness_tasks=[],
            tags=["audio"],
        )
        results.append(result)
    return results


# run this to clean up gs://marin-us-central1/tokenized/nemotron_cc/hq_actual-5af4cc/.executor_info
# from experiments.nemotron_cc.tokenize_nemotron import tokenize_nemotron_steps
# original_nemotron_tokenized = tokenize_nemotron_steps()["nemotron_cc/hq_actual"]

if __name__ == "__main__":
    executor_main(
        steps=run_data_mix_experiment(),
        # steps=[original_nemotron_tokenized],
        description="Experiment: Yodas2-En mixed with Nemotron",
    )
