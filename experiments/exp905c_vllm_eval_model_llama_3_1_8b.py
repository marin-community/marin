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

"""Run vLLM evals for Llama-3.1-8B (base)."""

from experiments.exp905c_vllm_eval_model import run_model_eval
from experiments.models import llama_3_1_8b


MODEL_CONFIG = {
    "name": "llama-3.1-8b",
    "path": "gs://marin-us-central2/gcsfuse_mount/models/meta-llama--Llama-3-1-8B--d04e592",
    "apply_chat_template": True,
    "tensor_parallel_size": 1,
}


if __name__ == "__main__":
    run_model_eval(MODEL_CONFIG, download_steps=[llama_3_1_8b])
