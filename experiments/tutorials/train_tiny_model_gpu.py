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
This is a tutorial on how to train a tiny model on a small dataset using a GPU.

This script demonstrates how to:
1. Train a tiny model on Wikitext-2 using a single GPU
2. Use GPU-specific training configuration
3. Run a quick training experiment

For CPU training, see train_tiny_model_cpu.py
"""

from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import executor_main, versioned

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama_nano
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig

# 1. Choose a dataset
wikitext_hf_id = "dlwh/wikitext_2_detokenized"

# For this tutorial, we limit to 1000 documents per shard
wikitext_tokenized = default_tokenize(
    name=wikitext_hf_id,
    dataset=wikitext_hf_id,
    tokenizer=marin_tokenizer,
    format=TextLmDatasetFormat(),
    sample_count=versioned(1000),
)


nano_train_config = SimpleTrainConfig(
    # Here we define the hardware resources we need.
    resources=ResourceConfig.with_gpu(count=1),
    train_batch_size=32,
    num_train_steps=100,
    learning_rate=6e-4,
    weight_decay=0.1,
)

nano_wikitext_model = default_train(
    name="llama-nano-wikitext",
    # Steps can depend on other steps: nano_wikitext_model depends on wikitext_tokenized
    tokenized=wikitext_tokenized,
    model_config=llama_nano,
    train_config=nano_train_config,
    tags=["llama", "nano", "wikitext", "tutorial"],
    # no point in running evals on such a tiny model
    eval_harness_tasks=[],
    use_default_validation=False,
)


if __name__ == "__main__":
    executor_main(
        steps=[
            wikitext_tokenized,
            nano_wikitext_model,
        ]
    )
