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

"""Run vLLM evals for OLMo3-7B-Base."""

from experiments.exp905c_vllm_eval_model import run_model_eval
from experiments.models import ModelConfig, download_model_step


MODEL_CONFIG = {
    "name": "olmo3-7b-base",
    "path": "gs://marin-us-central2/gcsfuse_mount/models/allenai--Olmo-3-1025-7B--main",
    "apply_chat_template": True,
    "tensor_parallel_size": 1,
}

DOWNLOAD_STEPS = [
    download_model_step(
        ModelConfig(
            hf_repo_id="allenai/Olmo-3-1025-7B",
            hf_revision="main",
        )
    )
]


if __name__ == "__main__":
    run_model_eval(MODEL_CONFIG, download_steps=DOWNLOAD_STEPS)
