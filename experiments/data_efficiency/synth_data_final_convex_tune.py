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

"""Convex-tune the best runs from each copy-scaling series (Simple, Stitched, Latent)."""

from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from experiments.data_efficiency.utils import get_synth_bounding_box
from marin.execution.executor import executor_main

# (synthetic_data_name, epochs_center, lr_center, wd_center, mix_ratio_center)
best_runs = [
    ("w2s", 4,  3e-3, 0.80, 0.75),
    ("s4",  8,  3e-3, 0.80, 0.75),
    ("s8",  8,  3e-3, 0.80, 0.75),
    ("s16", 16, 3e-3, 0.40, 0.75),
    ("s32", 16, 3e-3, 0.40, 0.75),
    ("w2",  8,  3e-3, 0.80, 0.75),
    ("b4",  8,  3e-3, 0.40, 0.75),
    ("b8",  16, 3e-3, 0.40, 0.75),
    ("b16", 16, 3e-3, 0.40, 0.75),

    ("b32", 32, 3e-3, 0.40, 0.90),
    ("b32", 32, 1e-3, 0.40, 0.90),
    
    ("z2",  4,  3e-3, 0.80, 0.75),
    ("z4",  8,  3e-3, 0.80, 0.75),
    ("z8",  8,  3e-3, 0.40, 0.75),
    ("z16", 16, 3e-3, 0.40, 0.75),
    ("z32", 32, 3e-3, 0.40, 0.90),
    ("z32", 32, 1e-3, 0.40, 0.90),
]

train_steps = []
for synthetic_data_name, epochs_center, lr_center, wd_center, mix_center in best_runs:
    for base_train_steps, model_name, epochs, lr, weight_decay, mix_ratio in get_synth_bounding_box(
        777, epochs_center, lr_center, wd_center, "300m4k", mix_center
    ):
        train_steps.append(
            data_efficiency_train_step(
                DataEfficiencyConfig(
                    data_name="dcr",
                    val_name=["dc_1k_val_normal"],
                    block_cross_document_attention=False,
                    epochs=epochs,
                    base_train_steps=base_train_steps,
                    train_batch_size=64,
                    lr=lr,
                    weight_decay=weight_decay,
                    train_seq_len=4096,
                    teacher_data_weights={synthetic_data_name: mix_ratio},
                    wandb_project_name="suhas-data-efficiency",
                    model_name=model_name,
                    tpu_type="v4-64",
                    nametag="",
                    wandb_additional_tags=["synth_data_efficiency", "cda", "convex_tune"],
                    initialize_from_hf=None,
                )
            )
        )

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Convex tune best synth data runs",
    )
