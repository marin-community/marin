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

"""Standard comparison runs for Mixtral MoE implementations."""

from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.custom_mixtral import MixtralConfig
from fray.v2 import ResourceConfig
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

AUTHOR = Author(name="Pranshu Chaturvedi", affiliation="Stanford University", url="https://stanford.edu/~pranshu")


def build_config(*, use_gmm: bool, name: str) -> SpeedrunConfig:
    cfg = MixtralConfig(
        seq_len=1024,
        hidden_dim=768,
        intermediate_dim=768,
        num_layers=12,
        num_heads=12,
        num_kv_heads=12,
        gradient_checkpointing=True,
        scan_layers=True,
        n_routed_experts=32,
        num_experts_per_tok=4,
        use_gmm=use_gmm,
        lbl_coef=None,
        rzl_coef=None,
    )
    return SpeedrunConfig(
        author=AUTHOR,
        description=f"Mixtral MoE comparison run ({name}).",
        model_config=cfg,
        train_config=SimpleTrainConfig(
            ResourceConfig.with_tpu("v4-8", slice_count=1),
            train_batch_size=256,
            num_train_steps=500,
            learning_rate=5e-4,
            weight_decay=0.1,
            steps_per_eval=250,
        ),
    )


def main():
    ragged = build_config(use_gmm=False, name="ragged-dot")
    gmm = build_config(use_gmm=True, name="grouped-matmul")
    executor_main(steps=default_speedrun("pranshu_mixtral_moe_compare_ragged", ragged))
    executor_main(steps=default_speedrun("pranshu_mixtral_moe_compare_gmm", gmm))


if __name__ == "__main__":
    main()
