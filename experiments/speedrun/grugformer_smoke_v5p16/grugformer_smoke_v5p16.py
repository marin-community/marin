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
Grugformer smoke test (TPU v5p-16).

Purpose: repeatedly-submittable short run to validate the Grugformer training stack on central1.

Pinned settings (requested):
- TPU: v5p-16
- Global batch size: 64
- Sequence length: 2048 (Splash attention compatible; multiple of 128)
- Dataset: speedrun defaults (FineWeb-Edu tokenized subcache + Paloma validation sets)
"""

# nodryrun

import os
import time
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from haliax import Axis
from jaxtyping import PRNGKeyArray

from experiments.defaults import default_train
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.prebuilt_caches import fineweb_edu_subcache_10B
from levanter.grug.model import GrugModelConfig
from levanter.models.grug_wrapper import GrugWrapper
from levanter.models.lm_model import LmConfig
from marin.execution.executor import executor_main


@LmConfig.register_subclass("grugformer_smoke_v5p16")
@dataclass(frozen=True)
class GrugformerSmokeConfig(LmConfig[GrugWrapper]):
    """LmConfig wrapper around canonical grug core hyperparameters."""

    # LmConfig field
    max_seq_len: int = 2048

    # Keep the model small-ish for smoke (compile + step time), but non-trivial.
    hidden_dim: int = 768
    intermediate_dim: int = 2688
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 12
    head_dim: int | None = None

    @property
    def model_type(self) -> type[GrugWrapper]:
        return GrugWrapper

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> GrugWrapper:
        grug_cfg = GrugModelConfig(
            vocab_size=Vocab.size,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
        )
        return GrugWrapper.init(Vocab, grug_cfg, key=key)


def main() -> None:
    # Requested fixed smoke settings.
    train_seq_len = 2048
    train_batch_size = 64
    num_train_steps = int(os.environ.get("SMOKE_STEPS", "5"))
    suffix = os.environ.get("SMOKE_RUN_SUFFIX")
    if not suffix:
        suffix = time.strftime("%Y%m%d_%H%M%S")

    model_cfg = GrugformerSmokeConfig(max_seq_len=train_seq_len)

    train = SimpleTrainConfig(
        ResourceConfig.with_tpu("v5p-16"),
        train_seq_len=train_seq_len,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=10_000,
        steps_per_hf_export=-1,
        explicit_mesh_axes=True,
    )

    # Use the speedrun-default tokenized dataset (FineWeb-Edu subcache) + default Paloma validations.
    train_step = default_train(
        name=f"smoke/grugformer_v5p16_b64_s2048_{suffix}",
        tokenized=fineweb_edu_subcache_10B,
        model_config=model_cfg,
        train_config=train,
        tags=["smoke", "grugformer", "v5p-16", "b64", "s2048"],
        eval_harness_tasks=(),
        use_default_validation=True,
    )

    executor_main(steps=[train_step], description="Grugformer smoke test (v5p-16, bs=64, seq=2048).")


if __name__ == "__main__":
    main()
