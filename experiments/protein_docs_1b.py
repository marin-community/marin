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

"""Train a ~1B parameter Qwen3 model on the protein-docs dataset."""

from levanter.data.text import TextLmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.cautious import CautiousConfig

from experiments.defaults import default_download, default_train
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize

TOKENIZER = "WillHeld/contactdoc-tokenizer"

qwen3_protein_docs_1b = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=1920,
    intermediate_dim=7680,
    num_heads=15,
    num_kv_heads=15,
    num_layers=19,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

optimizer_config = CautiousConfig(
    learning_rate=2.75e-3,
    weight_decay=0.1,
    min_lr_ratio=0.0,
    warmup=0.1,
    beta1=0.95,
    beta2=0.9604,
    epsilon=1e-15,
    max_grad_norm=1,
    adamc_weight_decay=True,
    lr_schedule="linear",
    decay=0.2,
)

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

protein_docs_download = default_download(
    name="raw/protein-docs",
    hf_dataset_id="timodonnell/protein-docs",
    revision="b3719628abb8bf3d7f02d8283cf37420c5146a4d",
)

protein_docs_tokenized = ExecutorStep(
    name="tokenized/protein-docs",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[protein_docs_download / "data/train-*.parquet"],
        validation_paths=[protein_docs_download / "data/validation-*.parquet"],
        cache_path=this_output_path(),
        tokenizer=versioned(TOKENIZER),
        format=TextLmDatasetFormat(text_key="document"),
    ),
)

protein_docs_data = lm_data_config(protein_docs_tokenized, permutation_type="feistel")

protein_docs_1b_model = default_train(
    name="protein-docs-1b",
    tokenized=protein_docs_data,
    model_config=qwen3_protein_docs_1b,
    train_config=train_config,
    tags=("protein-docs", "1b", "qwen3"),
    use_default_validation=False,
    eval_harness_tasks=[],
    override_output_path="checkpoints/protein-docs-1b",
)

if __name__ == "__main__":
    executor_main(steps=[protein_docs_1b_model])
