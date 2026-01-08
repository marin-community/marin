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

from fray.cluster import ResourceConfig

from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.evals.task_configs import EvalTaskConfig

from experiments.llama import llama3_tokenizer

from experiments.exp1342_gemstones_scaling_law import distributional_eval_sets
from experiments.isoflop_sweep import MARIN_2025_RECIPE, MARIN_SCALING_SUITES
from experiments.models import ModelConfig, download_model_step
from marin.execution.executor import executor_main, output_path_of, versioned
from marin.evaluation.log_probs import default_lm_log_probs
from marin.processing.tokenize import get_vocab_size_for_tokenizer

# Vocab size for building model configs
VOCAB_SIZE = get_vocab_size_for_tokenizer("stanford-crfm/marin-tokenizer")

# This is painfully slow to run in dry run mode
# nodryrun


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
    for model, candidate in list(zip(*MARIN_SCALING_SUITES["nemotron"], strict=False)):
        total_tokens = candidate.batch_size * candidate.train_steps * 4096
        name = (
            f"marin-nemo-{candidate.flops_budget:.0e}C-{total_tokens}T-"
            f"N{candidate.target_params:.0e}"
        )

        step = evaluate_levanter_lm_evaluation_harness(
            model_name=name,
            model_path=output_path_of(model),
            evals=tasks,
            resource_config=ResourceConfig.with_tpu("v5p-8"),
        )
        steps.append(step)

        model_config = MARIN_2025_RECIPE.build_model_config(candidate.target_params, VOCAB_SIZE)
        logprobs_step = default_lm_log_probs(
            output_path_of(model).cd("checkpoints"),
            model_config,
            dist_eval,
            resource_config=ResourceConfig.with_tpu("v5p-8"),
            checkpoint_is_hf=False,
            name=versioned(f"{name}-DistRobust-ICE-logprobs"),
        )

        steps.append(logprobs_step)

    for model, candidate in list(zip(*MARIN_SCALING_SUITES["common_pile"], strict=False)):
        total_tokens = candidate.batch_size * candidate.train_steps * 4096
        name = (
            f"marin-comma-{candidate.flops_budget:.0e}C-{total_tokens}T-"
            f"N{candidate.target_params:.0e}"
        )

        step = evaluate_levanter_lm_evaluation_harness(
            model_name=name,
            model_path=output_path_of(model),
            evals=tasks,
            resource_config=ResourceConfig.with_tpu("v5p-8"),
        )
        steps.append(step)

        model_config = MARIN_2025_RECIPE.build_model_config(candidate.target_params, VOCAB_SIZE)
        logprobs_step = default_lm_log_probs(
            output_path_of(model).cd("checkpoints"),
            model_config,
            dist_eval,
            resource_config=ResourceConfig.with_tpu("v5p-8"),
            checkpoint_is_hf=False,
            name=versioned(f"{name}-DistRobust-ICE-logprobs"),
        )

        steps.append(logprobs_step)

    for model, candidate in list(zip(*MARIN_SCALING_SUITES["dclm-default"], strict=False)):
        total_tokens = candidate.batch_size * candidate.train_steps * 4096
        name = (
            f"marin-dclm-{candidate.flops_budget:.0e}C-{total_tokens}T-"
            f"N{candidate.target_params:.0e}"
        )

        step = evaluate_levanter_lm_evaluation_harness(
            model_name=name,
            model_path=output_path_of(model),
            evals=tasks,
            resource_config=ResourceConfig.with_tpu("v5p-8"),
        )
        steps.append(step)

        model_config = MARIN_2025_RECIPE.build_model_config(candidate.target_params, VOCAB_SIZE)
        logprobs_step = default_lm_log_probs(
            output_path_of(model).cd("checkpoints"),
            model_config,
            dist_eval,
            resource_config=ResourceConfig.with_tpu("v5p-8"),
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
            resource_config=ResourceConfig.with_tpu("v5p-8"),
        )
        steps.append(step)
    return steps


if __name__ == "__main__":
    all_steps = create_eval_steps()
    executor_main(all_steps)
