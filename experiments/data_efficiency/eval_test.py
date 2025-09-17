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
Evaluate ensemble members specified in ensemble_members.py
"""

from experiments.data_efficiency.train import data_efficiency_eval_ensemble
from marin.execution.executor import executor_main
from experiments.data_efficiency.ensemble_members import ensemble_members_train_steps_dict

eval_steps = []

max_runs = 5

for key in ensemble_members_train_steps_dict:
    if key[-1] == 0:
        ensemble_members = []
        for seed in range(max_runs):
            key_copy = key[:-1] + (seed,)
            ensemble_members.append(ensemble_members_train_steps_dict[key_copy])

        for seed_count in range(1, max_runs + 1):
            eval_steps.append(
                data_efficiency_eval_ensemble(ensemble_members[:seed_count], key="varying-hparams-experiment")
            )

if __name__ == "__main__":
    executor_main(
        steps=eval_steps,
        description="Data scaling baseline",
    )
