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

"""Train a ~1B parameter Qwen3 model on the Tomol25 dataset.

Model configuration (~1.13B parameters with vocab_size=7,000):
- hidden_dim: 1920
- intermediate_dim: 7680 (1920 * 4)
- num_layers: 19
- num_heads: 15 (1920 // 128)
- num_kv_heads: 15

Training configuration:
- 50B tokens = 256 batch_size * 48,000 steps * 4096 seq_len
- Learning rate: 2.75e-3 = (0.33 * sqrt(256)) / 1920
- Beta2: 0.9604 = 0.98^(256/128)
"""

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.cautious import CautiousConfig

from experiments.defaults import default_download, default_train
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize

TOKENIZER = "WillHeld/marin-tomol"

# Model config (~1.13B params with vocab_size=7000)
# Derived from isoflop rules: hidden_dim=1920, layers=19, MLP_RATIO=4
qwen3_tomol_1b = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=1920,
    intermediate_dim=7680,  # 1920 * 4
    num_heads=15,  # 1920 // 128
    num_kv_heads=15,
    num_layers=19,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

# Optimizer config (isoflop patterns)
optimizer_config = CautiousConfig(
    learning_rate=2.75e-3,  # (0.33 * sqrt(256)) / 1920
    weight_decay=0.1,
    min_lr_ratio=0.0,
    warmup=0.1,
    beta1=0.95,
    beta2=0.9604,  # 0.98^(256/128)
    epsilon=1e-15,
    max_grad_norm=1,
    adamc_weight_decay=True,
    lr_schedule="linear",
    decay=0.2,
)

# Training config: 50B tokens = 256 * 48000 * 4096
train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v5p-8"),
    train_batch_size=256,
    num_train_steps=48_000,
    learning_rate=2.75e-3,
    weight_decay=0.1,
    min_lr_ratio=0.0,
    lr_schedule="linear",
    decay=0.2,
    warmup=0.1,
    optimizer_config=optimizer_config,
)

# Download dataset from HuggingFace
tomol_download = default_download(
    name="raw/tomol25",
    hf_dataset_id="WillHeld/Tomol25",
    revision="2087cc0ebe8379ab9962d52f9177c197d819c1c5",
)

# Tokenize with separate train/validation splits
tomol_tokenized = ExecutorStep(
    name="tokenized/tomol25",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[tomol_download / "data/train-*.parquet"],
        validation_paths=[tomol_download / "data/validation-*.parquet"],
        cache_path=this_output_path(),
        tokenizer=versioned(TOKENIZER),
    ),
)

tomol_data = lm_data_config(tomol_tokenized, permutation_type="feistel")

# Training step (use Tomol25's own validation split, not Paloma)
tomol_1b_model = default_train(
    name="tomol25-1b",
    tokenized=tomol_data,
    model_config=qwen3_tomol_1b,
    train_config=train_config,
    tags=("tomol", "1b", "qwen3"),
    use_default_validation=False,
    eval_harness_tasks=[],
    override_output_path="checkpoints/tomol25-1b",
)

if __name__ == "__main__":
    executor_main(steps=[tomol_1b_model])
