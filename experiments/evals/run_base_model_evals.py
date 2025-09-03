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
Base model evaluations across multiple LLMs.

This experiment evaluates OLMO Base 8B, LLAMA 3.1 8B, Deeper Starling 8B,
MAP-NEO 7B, and Amber Base 7B models on CORE_TASKS (augmented with OLMo Eval Tasks) as well
as dedicated MMLU 0-shot and 5-shot configurations.
"""

from experiments.evals.evals import default_base_eval
from experiments.models import amber_base_7b, llama_3_1_8b, map_neo_7b, olmo_2_base_8b
from marin.execution.executor import executor_main

if __name__ == "__main__":
    # Model path for deeper starling
    deeper_starling_path = "gs://marin-us-central2/checkpoints/tootsie-8b-deeper-starling/hf/step-1419999"
    # Run all evaluations on all models
    executor_main(
        steps=[
            *default_base_eval(deeper_starling_path),
            *default_base_eval(llama_3_1_8b),
            *default_base_eval(olmo_2_base_8b),
            *default_base_eval(amber_base_7b, engine_kwargs={"max_model_len": 2048, "max_gen_toks": 2048}),
            *default_base_eval(
                map_neo_7b, engine_kwargs={"trust_remote_code": True, "max_model_len": 4096, "max_gen_toks": 4096}
            ),
        ]
    )
