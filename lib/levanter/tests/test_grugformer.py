# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

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

from levanter.grugformer.config import AttentionRuntimeConfig, GrugModelConfig, GrugTrainingConfig
from levanter.grugformer.main import run_training


def test_synthetic_training_step_runs():
    cfg = GrugTrainingConfig(
        model=GrugModelConfig(
            vocab_size=257,
            hidden_dim=64,
            intermediate_dim=256,
            num_layers=1,
            num_heads=4,
            num_kv_heads=4,
            max_seq_len=16,
        ),
        attention=AttentionRuntimeConfig(backend="reference"),
        learning_rate=1e-3,
        weight_decay=0.01,
        steps=1,
        global_batch_size=2,
        seed=0,
    )

    run_training(cfg, cache_dir=None)
