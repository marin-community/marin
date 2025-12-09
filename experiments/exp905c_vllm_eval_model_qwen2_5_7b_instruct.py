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

"""Run vLLM evals for Qwen2.5-7B-Instruct."""

from experiments.exp905c_vllm_eval_model import MODEL_ROOT, run_model_eval


MODEL_CONFIG = {
    "name": "qwen2.5-7b",
    "path": f"{MODEL_ROOT}/qwen2.5-7b",
    "apply_chat_template": True,
    "max_model_len": int(8094 * 4),
    "tensor_parallel_size": 4,
}


if __name__ == "__main__":
    run_model_eval(MODEL_CONFIG)
