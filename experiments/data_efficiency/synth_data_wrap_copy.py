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


from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from experiments.data_efficiency.utils import get_bounding_box
from marin.execution.executor import executor_main

val_name_dict = {
    "dcr": ["dc_t10x_val", "dc_t10x_val_shuffled", "dc_1k_val_normal"],
    "b32": ["dc_1k_val_normal"],
    "s32": ["dc_1k_val_normal"],
    "b16": ["dc_1k_val_normal"],
    "s16": ["dc_1k_val_normal"],
}

train_steps = [
    [
        data_efficiency_train_step(
            DataEfficiencyConfig(
                data_name=train_data_name,
                val_name=val_name_dict[train_data_name],
                teacher_data_name=synthetic_data_name,
                teacher_data_weight=synthetic_data_weight,
                block_cross_document_attention=block_cross_document_attention,
                epochs=epochs,
                base_train_steps=base_train_steps,
                train_batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                train_seq_len=train_seq_len,
                wandb_project_name="suhas-data-efficiency",
                model_name=model_name,
                tpu_type="v4-64",
                nametag=nametag,
                wandb_additional_tags=["synth_data_efficiency", "effective_context"] + (["cda"] if not block_cross_document_attention else []),
                initialize_from_hf=None,
            )
        )

    ###### Pick one of convex mode or standard mode ######
    # ## Convex mode
    #   for base_train_steps, epochs, lr, weight_decay, model_name in get_bounding_box(base_train_steps_center, epochs_center, lr_center, weight_decay_center, model_name_center)
    # ]
    # for base_train_steps_center, epochs_center, lr_center, weight_decay_center, model_name_center, batch_size, train_seq_len in [

    ## Standard mode
    ]
    for base_train_steps, epochs, lr, weight_decay, model_name, batch_size, train_seq_len in [

    ###### Pick which data to train with ######
    ## Wrap ICPT
        (777, 16, 3e-3, 0.4, "300m4k", 64, 4096),
        (777, 16, 3e-3, 0.8, "300m4k", 64, 4096),
        # (777, 16, 3e-3, 1.6, "300m4k", 64, 4096),
        (777, 32, 3e-3, 0.4, "300m4k", 64, 4096),
        (777, 32, 3e-3, 0.8, "300m4k", 64, 4096),
    ]
    for synthetic_data_name, synthetic_data_weight in [
        ("b32", 0.75),
        # ("b32", 0.9),
        ("s32", 0.75),
        # "s32", 0.9),
    ]
    for train_data_name in ["dcr"]
    for block_cross_document_attention, nametag in [
        (False, ""),
        # (True, ""),
    ]
]

train_steps = [step for sublist in train_steps for step in sublist]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Data scaling baseline",
    )
