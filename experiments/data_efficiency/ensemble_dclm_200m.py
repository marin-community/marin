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
Train a collection of ensemble members for hyper-parameters of interest
"""

from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from marin.execution.executor import executor_main

ensemble_members_train_steps_dict_list = [
    {
        (base_train_steps, epochs, lr, weight_decay, model_name, seed): data_efficiency_train_step(
            DataEfficiencyConfig(
                train_seed=seed,
                data_seed=seed,
                data_name="dclm_200m",
                val_name="dclm_200m_val",
                epochs=epochs,
                base_train_steps=base_train_steps,
                train_batch_size=64,
                lr_schedule="cosine",
                lr=lr,
                weight_decay=weight_decay,
                wandb_project_name="suhas-data-efficiency",
                model_name=model_name,
                nametag=f"-seed{seed}",
                tpu_type="v4-64",
            )
        )
        for base_train_steps, epochs, lr, weight_decay, model_name in [candidate_hparams]
    }
    for candidate_hparams in [
        # # Optimal regularized hyper-parameters
        # (750, 8, 3e-3, 0.8, "150m4k"),
        # (750, 16, 3e-3, 1.6, "300m4k"),
        # (750, 8, 1e-3, 3.2, "600m4k"),
        # (750, 8, 1e-3, 3.2, "1_4b4k"),
        # (750, 8, 1e-3, 3.2, "1_5b4k"),
        # # Ensembling guess
        # (750, 32, 3e-3, 0.4, "150m4k"),
        # (750, 32, 3e-3, 0.8, "300m4k"),
        (750, 16, 1e-3, 1.6, "600m4k"),
        (750, 16, 1e-3, 3.2, "1_5b4k"),
        (750, 16, 1e-3, 1.6, "1_5b4k"),
    ]
    for seed in list(range(5))
]

ensemble_members_train_steps_dict = {}
for train_steps_dict in ensemble_members_train_steps_dict_list:
    for key, value in train_steps_dict.items():
        assert key not in ensemble_members_train_steps_dict
        ensemble_members_train_steps_dict[key] = value

ensemble_members_train_steps = list(ensemble_members_train_steps_dict.values())

if __name__ == "__main__":
    executor_main(
        steps=ensemble_members_train_steps,
        description="Ensemble baseline",
    )
