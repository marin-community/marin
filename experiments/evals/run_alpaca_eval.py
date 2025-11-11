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

# nodryrun
from experiments.evals.engine_configs import DEFAULT_VLLM_ENGINE_KWARGS
from experiments.evals.evals import evaluate_alpaca_eval
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.execution.executor import ExecutorMainConfig, executor_main

executor_main_config = ExecutorMainConfig(force_run_failed=True)
steps = [
    evaluate_alpaca_eval(
        # model_name="debug_llama3_tulu_5e-6_3eps_fix_tokenizer",
        # model_path="gs://marin-us-central2/checkpoints/train_lm_llama3_tulu_sft/hf/seed_0/ivl3pggc/step-3833/",
        model_name="debug_tootsie_tulu_lr1e-4",
        model_path="gs://marin-us-central2/checkpoints/train_lm_tootsie_tulu_lr1e-4/hf/seed_0/x5f4exqk/step-3833",
        resource_config=SINGLE_TPU_V6E_8,
        engine_kwargs=DEFAULT_VLLM_ENGINE_KWARGS,
        # llama 3
        stop_token_ids=[128009],
        # olmo
        # stop_token_ids=[100257],
    ),
]

if __name__ == "__main__":
    executor_main(executor_main_config, steps=steps)
