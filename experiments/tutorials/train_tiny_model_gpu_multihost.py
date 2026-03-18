# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a tiny model across 2 GPU hosts — used for multi-host canary smoke tests.

Same as train_tiny_model_gpu.py but with replicas=2 for multi-host.
"""

from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import executor_main, versioned

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama_nano
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig

wikitext_hf_id = "dlwh/wikitext_2_detokenized"

wikitext_tokenized = default_tokenize(
    name=wikitext_hf_id,
    dataset=wikitext_hf_id,
    tokenizer=marin_tokenizer,
    format=TextLmDatasetFormat(),
    sample_count=versioned(1000),
)

nano_train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_gpu("H100", count=8, cpu=32, disk="128G", ram="128G", replicas=2),
    train_batch_size=256,
    num_train_steps=10,
    learning_rate=6e-4,
    weight_decay=0.1,
)

nano_wikitext_model = default_train(
    name="llama-nano-wikitext-multihost",
    tokenized=wikitext_tokenized,
    model_config=llama_nano,
    train_config=nano_train_config,
    tags=["llama", "nano", "wikitext", "tutorial", "multihost"],
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
