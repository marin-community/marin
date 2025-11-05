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

from dataclasses import dataclass


@dataclass
class ResourceConfig:
    num_tpu: int
    tpu_type: str
    strategy: str
    include_head_in_scheduling_strategy: bool = False


SINGLE_TPU_V4_8 = ResourceConfig(num_tpu=1, tpu_type="TPU-v4-8", strategy="STRICT_PACK")  # us-central2
SINGLE_TPU_V4_16 = ResourceConfig(num_tpu=2, tpu_type="TPU-v4-16", strategy="STRICT_PACK")  # us-central2
SINGLE_TPU_V5p_8 = ResourceConfig(num_tpu=1, tpu_type="TPU-v5p-8", strategy="STRICT_PACK")  # us-central1
SINGLE_TPU_V5p_8_FULL = ResourceConfig(
    num_tpu=4, tpu_type="TPU-v5p-8", strategy="STRICT_PACK", include_head_in_scheduling_strategy=True
)  # us-central1
SINGLE_TPU_V6E_8 = ResourceConfig(num_tpu=1, tpu_type="TPU-v6e-8", strategy="STRICT_PACK")  # us-east5
TPU_V6E_8_STRICT_PACK = ResourceConfig(num_tpu=8, tpu_type="TPU-v6e-8", strategy="STRICT_PACK")  # us-east5
TPU_V4_16_STRICT_PACK = ResourceConfig(num_tpu=8, tpu_type="TPU-v4-16", strategy="PACK")  # us-central2
