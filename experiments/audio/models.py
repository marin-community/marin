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

from experiments.models import ModelConfig, download_model_step

qwen3x_0_6b_base = download_model_step(
    ModelConfig(
        hf_repo_id="potsawee/qwen3x-0.6B-base",
        hf_revision="1a84125",
    )
)

qwen3x_1_7b_base = download_model_step(
    ModelConfig(
        hf_repo_id="potsawee/qwen3x-1.7B-base",
        hf_revision="14cd3a1",
    )
)

soda_600m_base = download_model_step(
    ModelConfig(
        hf_repo_id="potsawee/soda-600m-base",
        hf_revision="86c9e30",
    )
)

soda_600m_warmstart = download_model_step(
    ModelConfig(
        hf_repo_id="potsawee/soda-600m-warmstart",
        hf_revision="df960e9",
    )
)

# SODA-600M preliminary experiment, but this model was trained multilingual
# Yodas: en, de, fr, es, th, hi, ar, zh
blueberry_600m = download_model_step(
    ModelConfig(
        hf_repo_id="WillHeld/blueberry",
        hf_revision="9dc7905",
    )
)
