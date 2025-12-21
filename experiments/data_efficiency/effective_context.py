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
Script to run the neighborhood of a given hyper-parameter to search for locally-optimal hyper-parameters
"""

from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from experiments.data_efficiency.utils import get_bounding_box
from marin.execution.executor import executor_main

# 800 steps ==> 200M tokens
# 1600 steps ==> 400M tokens
# 3200 steps ==> 800M tokens
# 6400 steps ==> 1600M tokens

val_name_dict = {
    "dclm_200m_sorted": ["dclm_200m_shuffled_val", "dclm_200m_sorted_val"],
    "dclm_200m_shuffled": ["dclm_200m_sorted_val", "dclm_200m_shuffled_val"],
    "mn_lc": ["mn_lc_val", "mn_split_val"],
    "mn_split": ["mn_lc_val", "mn_split_val"],
    "dclm_tsp": ["dclm_tsp_val", "dclm_tsp_val_shuffled"],
    "dclm_shuffled": ["dclm_200m_sorted_val", "dclm_200m_shuffled_val", "dclm_tsp_val", "dclm_tsp_val_shuffled"],
    "dclm": None,
    "dc_t10x": ["dc_t10x_val", "dc_t10x_val_shuffled"],
    "dc_t10x_shuffled": ["dc_t10x_val", "dc_t10x_val_shuffled"],
    "dc_shuffled": ["dc_t10x_val", "dc_t10x_val_shuffled"],
    "dc_1m": ["dc_1k_val_normal", "dc_t10x_val", "dc_t10x_val_shuffled"],
    "dc_1m_mix": ["dc_1k_val_normal", "dc_t10x_val", "dc_t10x_val_shuffled"],
    "dc_1_3m": ["dc_1k_val_normal", "dc_t10x_val", "dc_t10x_val_shuffled"],
    "dc_1_3m_mix": ["dc_1k_val_normal", "dc_t10x_val", "dc_t10x_val_shuffled"],
}

train_steps = [
    [
        data_efficiency_train_step(
            DataEfficiencyConfig(
                data_name=train_data_name,
                val_name=val_name_dict[train_data_name],
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
                wandb_additional_tags=["effective_context"] + (["cda"] if not block_cross_document_attention else []),
            )
        )

    ###### Pick one of convex mode or standard mode ######
    # ## Convex mode
    #    for base_train_steps, epochs, lr, weight_decay, model_name in get_bounding_box(base_train_steps_center, epochs_center, lr_center, weight_decay_center, model_name_center)
    # ]
    # for base_train_steps_center, epochs_center, lr_center, weight_decay_center, model_name_center, batch_size, train_seq_len in [

    ## Standard mode
    ]
    for base_train_steps, epochs, lr, weight_decay, model_name, batch_size, train_seq_len in [

    ###### Pick which data to train with ######
    ## DCLM
        (750, 16, 3e-3, 1.6, "300m4k", 1024, 256),
        (750, 16, 3e-3, 1.6, "300m4k", 512, 512),
        (750, 16, 3e-3, 1.6, "300m4k", 256, 1024),
        (750, 16, 3e-3, 1.6, "300m4k", 128, 2048),
        # (750, 16, 3e-3, 1.6, "300m4k", 64, 4096),
        # (750, 16, 3e-3, 1.6, "300m16k", 32, 8192),
        # (750, 16, 3e-3, 1.6, "300m16k", 16, 16384),
        # (6400, 8, 1e-3, 0.8, "1_5b4k", 64, 4096),
        # (6000, 8, 1e-3, 0.8, "1_5b4k", 64, 4096),
    ]
    for train_data_name in ["dc_t10x", "dc_t10x_shuffled"]
    # for train_data_name in ["dc_1_3m", "dc_1_3m_mix"]
    # for train_data_name in ["dclm_shuffled"]
    for block_cross_document_attention, nametag in [
        # (False, "-tspv"),
        (False, ""),
        (True, ""),
    ]

    # ## Multi-News
    #     (400, 16, 3e-3, 1.6, "300m4k", 64, 4096),
    #     # (400, 8, 3e-3, 1.6, "300m4k", 64, 4096),
    # ]
    # for train_data_name, block_cross_document_attention, nametag in [
    #     # ("mn_lc", False, "-e2"),
    #     ("mn_lc", True, ""),
    # ]
]

train_steps = [step for sublist in train_steps for step in sublist]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Data scaling baseline",
    )
