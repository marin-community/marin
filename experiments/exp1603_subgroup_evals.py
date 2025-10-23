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

from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.evals.resource_configs import SINGLE_TPU_V5p_8
from experiments.evals.task_configs import EvalTaskConfig

from experiments.llama import llama3_tokenizer

from experiments.exp1342_gemstones_scaling_law import distributional_eval_sets
from experiments.isoflop_sweep import MARIN_SCALING_SUITES
from experiments.models import ModelConfig, download_model_step
from marin.execution.executor import executor_main, output_path_of, versioned
from marin.evaluation.log_probs import default_lm_log_probs


def create_eval_steps() -> list:
    tasks = (
        EvalTaskConfig("anthropic_ai_risk", num_fewshot=0, task_alias="anthropic_ai_risk_delim2"),
        EvalTaskConfig("mmlu_sl_verb", num_fewshot=5, task_alias="mmlu_sl_verb_5_shot"),
        EvalTaskConfig("mmlu_sl", num_fewshot=5, task_alias="mmlu_sl_5_shot"),
        EvalTaskConfig("mmlu", num_fewshot=5, task_alias="mmlu_5_shot"),
        EvalTaskConfig("mmlu_continuation", num_fewshot=5, task_alias="mmlu_continuation_5_shot"),
    )

    steps = []
    dist_eval = distributional_eval_sets(llama3_tokenizer)
    for model, metadata in list(zip(*MARIN_SCALING_SUITES["nemotron"], strict=False)):
        name = f"marin-nemo-{metadata[0]}C-{metadata[-3] * metadata[-2] * 4096}T-{metadata[1]}W-{metadata[2]}D"

        step = evaluate_levanter_lm_evaluation_harness(
            model_name=name,
            model_path=output_path_of(model),
            evals=tasks,
            resource_config=SINGLE_TPU_V5p_8,
        )
        steps.append(step)

        logprobs_step = default_lm_log_probs(
            output_path_of(model).cd("checkpoints"),
            metadata[-1],
            dist_eval,
            resource_config=SINGLE_TPU_V5p_8,
            checkpoint_is_hf=False,
            name=versioned(f"{name}-DistRobust-ICE-logprobs"),
        )

        steps.append(logprobs_step)

    for model, metadata in list(zip(*MARIN_SCALING_SUITES["common_pile"], strict=False)):
        name = f"marin-comma-{metadata[0]}C-{metadata[-3] * metadata[-2] * 4096}T-{metadata[1]}W-{metadata[2]}D"

        step = evaluate_levanter_lm_evaluation_harness(
            model_name=name,
            model_path=output_path_of(model),
            evals=tasks,
            resource_config=SINGLE_TPU_V5p_8,
        )
        steps.append(step)

        logprobs_step = default_lm_log_probs(
            output_path_of(model).cd("checkpoints"),
            metadata[-1],
            dist_eval,
            resource_config=SINGLE_TPU_V5p_8,
            checkpoint_is_hf=False,
            name=versioned(f"{name}-DistRobust-ICE-logprobs"),
        )

        steps.append(logprobs_step)

    for model, metadata in list(zip(*MARIN_SCALING_SUITES["dclm-default"], strict=False)):
        name = f"marin-dclm-{metadata[0]}C-{metadata[-3] * metadata[-2] * 4096}T-{metadata[1]}W-{metadata[2]}D"

        step = evaluate_levanter_lm_evaluation_harness(
            model_name=name,
            model_path=output_path_of(model),
            evals=tasks,
            resource_config=SINGLE_TPU_V5p_8,
        )
        steps.append(step)

    logprobs_step = default_lm_log_probs(
        output_path_of(model).cd("checkpoints"),
        metadata[-1],
        dist_eval,
        resource_config=SINGLE_TPU_V5p_8,
        checkpoint_is_hf=False,
        name=versioned(f"{name}-DistRobust-ICE-logprobs"),
    )

    steps.append(logprobs_step)

    baselines = [
        ("allenai/OLMo-2-1124-7B", "stage2-ingredient3-step8000-tokens34B"),
        ("allenai/OLMo-2-1124-13B", "stage2-ingredient4-step9000-tokens76B"),
        ("meta-llama/Llama-3.2-1B", "main"),
        ("meta-llama/Llama-3.2-3B", "main"),
        ("meta-llama/Llama-3.1-8B", "main"),
        ("marin-community/marin-8b-base", "main"),
        ("meta-llama/Llama-2-7b-hf", "main"),
        ("meta-llama/Llama-2-13b-hf", "main"),
        ("Qwen/Qwen3-14B-Base", "main"),
        ("Qwen/Qwen3-8B-Base", "main"),
        ("Qwen/Qwen3-4B-Base", "main"),
        ("Qwen/Qwen3-1.7B-Base", "main"),
        ("Qwen/Qwen3-0.6B-Base", "main"),
    ]
    for model, revision in baselines:
        model_instance = download_model_step(ModelConfig(hf_repo_id=model, hf_revision=revision))

        step = evaluate_levanter_lm_evaluation_harness(
            model_name=f"{model}@{revision}",
            model_path=output_path_of(model_instance),
            evals=tasks,
            resource_config=SINGLE_TPU_V5p_8,
        )
        steps.append(step)
    return steps


if __name__ == "__main__":
    all_steps = create_eval_steps()
    executor_main(all_steps)
