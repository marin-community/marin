# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
This is a hello-world speedrun that trains a tiny model, and is designed to run on a single **GPU**.
See https://github.com/marin-community/marin/tree/main/docs/how-to-guides/submitting-speedrun.md for
further documentation around submitting speedrun, and for a more detailed walkthrough of the code
and process.
"""

import logging

from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

tiny_llama = LlamaConfig(
    max_seq_len=512,
    hidden_dim=16,
    intermediate_dim=64,
    num_heads=1,
    num_kv_heads=1,
    num_layers=1,
    tie_word_embeddings=True,
)

speedrun_config = SpeedrunConfig(
    author=Author(
        name="yun1104",
        affiliation="HuaZhong University of Science and Technology",
        url="https://github.com/yun1104/marin",
    ),
    description="Nano model based on Llama architecture. To run the first experiment, I need to use a small model. I use the RTX3090 GPU.",
    model_config=tiny_llama,
    train_config=SimpleTrainConfig(
        ResourceConfig.with_gpu("RTX3090", count=1),
        train_batch_size=32,
        num_train_steps=100,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
    ),
    # tokenized_dataset need not be set here; by default it will be set to a FineWeb-Edu 10B token dataset
    # that we have pre-tokenized and shuffled, available at https://huggingface.co/datasets/marin-community/fineweb-edu-pretokenized-10B
    # tokenized_dataset=fineweb_edu_subcache_10B,
)

# Shows your speedrun configuration, model FLOPs, model size and (training) hardware FLOPs- you
# can use this before actually kicking off a run to validate your setup
speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_nano_gpu_speedrun", speedrun_config))
