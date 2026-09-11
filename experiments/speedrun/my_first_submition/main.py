# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import experiment_main
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint

from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.prebuilt_caches import fineweb_edu_10B_dataset
from experiments.marin_tokenizer import marin_tokenizer

tiny_llama = LlamaConfig(
    max_seq_len=512,
    hidden_dim=16,
    intermediate_dim=64,
    num_heads=1,
    num_kv_heads=1,
    num_layers=1,
    tie_word_embeddings=True,
)


def build() -> ArtifactStep[LevanterCheckpoint]:
    return train_lm(
        name="checkpoints/llama-nano-gpu-speedrun",
        run_id="llama_nano_gpu_speedrun",
        model=tiny_llama,
        optimizer=AdamConfig(learning_rate=3e-3, weight_decay=0.1),
        datasets={fineweb_edu_10B_dataset(): 1.0},
        validation=list(paloma_datasets(tokenizer=marin_tokenizer).values()),
        batch_size=32,
        seq_len=tiny_llama.max_seq_len,
        num_train_steps=100,
        z_loss_weight=None,
        evals=None,
        resources=ResourceConfig.with_gpu("RTX3090", count=1),
        steps_per_eval=500,
        tags=["llama", "speedrun", "rtx3090"],
    )


if __name__ == "__main__":
    experiment_main(build)()
