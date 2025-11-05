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
Base SFT evaluations across multiple LLMs.

This experiment evaluates OLMO Base 8B, LLAMA 3.1 8B, Deeper Starling 8B,
MAP-NEO 7B, and Amber Base 7B models on CORE_TASKS (augmented with OLMo Eval Tasks) as well
as dedicated MMLU 0-shot and 5-shot configurations.
"""

from experiments.evals.evals import default_sft_eval
from experiments.models import llama_3_1_8b_instruct, tulu_3_1_8b_instruct
from experiments.tootsie.exp1237_starling_sft import mixture_sft_deeper_starling
from marin.execution.executor import executor_main

if __name__ == "__main__":
    # Run all evaluations on all models
    executor_main(
        steps=[
            *default_sft_eval(mixture_sft_deeper_starling),
            *default_sft_eval(llama_3_1_8b_instruct),
            *default_sft_eval(tulu_3_1_8b_instruct),
        ]
    )
