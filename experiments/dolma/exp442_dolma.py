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
Train Dolma/OLMo models.
https://github.com/marin-community/marin/issues/442
"""

from experiments.defaults import default_train
from experiments.llama import llama_1_4b, llama_1_4b_train_config
from experiments.pretraining_datasets import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

EXPERIMENT_TAG = ["442_dolma"]

dolma_llama3_tokenized = lm_mixture_data_config(
    components=tokenize_dolma(),
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
    permutation_type="linear",
)

dolma_1_4b = default_train(
    name="dolma-1.4b",
    tokenized=dolma_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

## olmo replications

# (neox is close enough to olmo tokenizer)
dolma_neox_tokenized = lm_mixture_data_config(
    components=tokenize_dolma(tokenizer="EleutherAI/gpt-neox-20b"),
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
    permutation_type="linear",
)

# https://arxiv.org/pdf/2402.00838 page 3 (Table 1
olmoish_1b_config = LlamaConfig(
    num_layers=16,
    hidden_dim=2048,
    intermediate_dim=7168,
    num_heads=16,
    num_kv_heads=16,
    max_seq_len=2048,
    tie_word_embeddings=True,
    # they don't learn the layer norm weights
    use_layer_norm_weight=False,
)

olmoish_1b_train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v5litepod-256", slice_count=1),
    learning_rate=4e-4,
    warmup=2000,
    weight_decay=0.1,
    train_batch_size=2048,
    num_train_steps=500000,  # 2048 * 2048 * 500000 = 2.1T tokens
)

olmoish_1b = default_train(
    name="olmoish-1b",
    tokenized=dolma_neox_tokenized,
    model_config=olmoish_1b_config,
    train_config=olmoish_1b_train_config,
    tags=[*EXPERIMENT_TAG, "olmoish", "1b"],
)


if __name__ == "__main__":
    executor_main(steps=[olmoish_1b, dolma_1_4b])
