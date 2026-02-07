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

# from experiments.models import qwen3_0_6b as qwen3_0_6b_base_download_model_step
# from experiments.models import qwen3_0_6b_base, qwen3_1_7b_base
from experiments.audio.models import qwen3x_0_6b_base, qwen3x_1_7b_base
from marin.execution.executor import executor_main

if __name__ == "__main__":
    executor_main(
        # steps=[qwen3_0_6b_base_download_model_step],
        # steps=[qwen3_0_6b_base, qwen3_1_7b_base],
        steps=[qwen3x_0_6b_base, qwen3x_1_7b_base],
        description="Download Qwen3 model weights for warm-start experiments.",
    )
