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

import ray


@ray.remote(resources={"TPU-v4-8-head": 1}, env_vars={"VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
def run_vllm():
    from vllm import LLM

    llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct", max_model_len=1024)
    outputs = llm.generate(["Hello, how are you?"])
    return outputs


if __name__ == "__main__":
    outputs = ray.get(run_vllm.remote())
    print(outputs)
