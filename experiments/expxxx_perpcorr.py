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

from experiments.llama import llama_3_2_1b as llama_3_2_1b_config, llama3_tokenizer, llama_8b
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import mixture_for_evaluation
from experiments.paloma import paloma_tokenized

from dataclasses import dataclass
from levanter.models.llama import LmConfig
from levanter.models.olmo import Olmo2Config
from experiments.evals.resource_configs import SINGLE_TPU_V5p_8_FULL
from experiments.evals.task_configs import EvalTaskConfig
from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from marin.execution.executor import ExecutorStep
from experiments.models import (
    llama_3_1_8b,
    olmo_2_base_8b,
    marin_8b_base,
    llama_3_2_1b as llama_3_2_1b_model,
    qwen3_0_6b,
    qwen3_1_7b,
    qwen3_4b,
    qwen3_8b,
    qwen3_0_6b_base,
    qwen3_1_7b_base,
    qwen3_4b_base,
    qwen3_8b_base,
)
from experiments.qwen3 import (
    qwen3_0_6b as qwen3_0_6b_config,
    qwen3_1_7b as qwen3_1_7b_config,
    qwen3_4b as qwen3_4b_config,
    qwen3_8b as qwen3_8b_config,
)
from experiments.isoflop_sweep import IsoFlopSweepConfig, generate_isoflop_steps
from experiments.tootsie.exp1295_32b import nemotron_mix
from experiments.uncheatable_eval import uncheatable_eval_tokenized

olmo_7b = Olmo2Config(
    seq_len=4096,
    hidden_dim=4096,
    intermediate_dim=11008,
    num_heads=32,
    num_kv_heads=32,
    num_layers=32,
)


@dataclass
class ModelConfig:
    model_name: str
    model_config: LmConfig
    tokenizer: str
    model_path: ExecutorStep


# Evaluate log probabilities of meta-llama/Llama-3.2-1B on a subset of DCLM baseline
# Uses 1024 samples by default (adjust max_samples_per_dataset as needed)

model_with_config = [
    ModelConfig(
        model_name="marin-community/marin-8b-base",
        model_config=llama_8b,
        tokenizer=llama3_tokenizer,
        model_path=marin_8b_base,
    ),
    ModelConfig(
        model_name="meta-llama/Llama-3.1-8B", model_config=llama_8b, tokenizer=llama3_tokenizer, model_path=llama_3_1_8b
    ),
    ModelConfig(
        model_name="meta-llama/Llama-3.2-1B",
        model_config=llama_3_2_1b_config,
        tokenizer=llama3_tokenizer,
        model_path=llama_3_2_1b_model,
    ),
    ModelConfig(
        model_name="allenai/OLMo-2-1124-7B",
        model_config=olmo_7b,
        tokenizer="allenai/OLMo-2-1124-7B",
        model_path=olmo_2_base_8b,
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-0.6B", model_config=qwen3_0_6b_config, tokenizer="Qwen/Qwen3-0.6B", model_path=qwen3_0_6b
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-1.7B", model_config=qwen3_1_7b_config, tokenizer="Qwen/Qwen3-1.7B", model_path=qwen3_1_7b
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-4B", model_config=qwen3_4b_config, tokenizer="Qwen/Qwen3-4B", model_path=qwen3_4b
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-8B", model_config=qwen3_8b_config, tokenizer="Qwen/Qwen3-8B", model_path=qwen3_8b
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-0.6B-Base",
        model_config=qwen3_0_6b_config,
        tokenizer="Qwen/Qwen3-0.6B",
        model_path=qwen3_0_6b_base,
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-1.7B-Base",
        model_config=qwen3_1_7b_config,
        tokenizer="Qwen/Qwen3-1.7B",
        model_path=qwen3_1_7b_base,
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-4B-Base",
        model_config=qwen3_4b_config,
        tokenizer="Qwen/Qwen3-4B",
        model_path=qwen3_4b_base,
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-8B-Base",
        model_config=qwen3_8b_config,
        tokenizer="Qwen/Qwen3-8B",
        model_path=qwen3_8b_base,
    ),
]


def get_directory_friendly_name(model_name: str) -> str:
    return model_name.replace("/", "--").replace(".", "-")


EVAL_TASKS = [
    # EvalTaskConfig("mmlu_sl_verb", num_fewshot=5, task_alias="mmlu_sl_verb_5_shot"),
    # EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot"),  # 4-way MCQ commonsense reasoning dataset,
    # EvalTaskConfig("gsm8k_loss", num_fewshot=8, task_alias="gsm8k_loss_8shot"),
    # EvalTaskConfig("math_500_loss", num_fewshot=0),
    ### NEW
    # EvalTaskConfig("mmlu_sl_verb", num_fewshot=5, task_alias="mmlu_sl_verb_5_shot"),
    # EvalTaskConfig(
    #     "hellaswag", 10, task_alias="hellaswag_10shot"
    # ),  # 4-way MCQ commonsense reasoning dataset,
    # EvalTaskConfig("gsm8k_loss", num_fewshot=8, task_alias="gsm8k_loss_8shot"),
    # EvalTaskConfig("math_500_loss", num_fewshot=0),
    # generation tasks
    EvalTaskConfig("gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot_8shot"),
    EvalTaskConfig("minerva_math", num_fewshot=4, task_alias="minerva_math_4shot"),
    EvalTaskConfig("mbpp", num_fewshot=3, task_alias="mbpp_3shot"),
    EvalTaskConfig("humaneval", num_fewshot=0, task_alias="humaneval_0shot"),
    EvalTaskConfig("bbh_cot_fewshot", num_fewshot=3, task_alias="bbh_cot_3shot"),
    # EvalTaskConfig("arc_easy", 10),  # 10-shot, four-way MCQ questions involving grade 3-9 basic science
    # EvalTaskConfig("arc_challenge", 10),  # a (harder) version of arc_easy
    # EvalTaskConfig("boolq", 10),  # answer yes/no questions based on a passage
    # EvalTaskConfig("commonsense_qa", 10),  # 5-way multiple-choice questions based on common-sense, everyday scenarios
    # EvalTaskConfig("copa", 0),  # use causal reasoning to predict the correct outcome of a given scenario
    # EvalTaskConfig("openbookqa", 0),  # 4-way multiple choice question answering task that
    # # requires multi-step reasoning
    # EvalTaskConfig("piqa", 10),  # answer questions based on a passage
    # (requires generation which is not supported in Levanter at the moment)
    # EvalTaskConfig("squadv2", 10),  # reading comprehension benchmark
    # EvalTaskConfig("winogrande", 0),  # Winograd challenge, extended to more domains)
    # EvalTaskConfig("sciq", 0),
    # EvalTaskConfig("social_iqa", 0),
    # EvalTaskConfig("race", 0, task_alias="race_high_0shot"),
]

steps = []
# isoflop_steps, isoflop_model_configs, isoflop_train_configs, isoflop_budgets, _ = generate_isoflop_sweep(
#     nemotron_mix,
#     experiment_name="nemo-wider-depth-adapt",
# )

sweep_cfg = IsoFlopSweepConfig(tokenized_dataset=nemotron_mix)
isoflop_steps, isoflop_model_configs, isoflop_train_configs, isoflop_budgets, _ = generate_isoflop_steps(
    sweep_cfg, "nemo-wider-depth-adapt"
)
for isoflop_step, isoflop_model_config, isoflop_train_config, isoflop_budget in zip(
    isoflop_steps, isoflop_model_configs, isoflop_train_configs, isoflop_budgets, strict=False
):
    experiment_name = isoflop_step.name.split("/")[-1]
    paloma_tokenized_dict = paloma_tokenized(tokenizer=llama3_tokenizer)
    uncheatable_eval_tokenized_dict = uncheatable_eval_tokenized(tokenizer=llama3_tokenizer)
    eval_data = mixture_for_evaluation(paloma_tokenized_dict | uncheatable_eval_tokenized_dict)

    wandb_tags = (
        f"FLOPs={isoflop_budget:.1e}",
        f"d={isoflop_model_config.hidden_dim}",
        f"L={isoflop_model_config.num_layers}",
        f"B={isoflop_train_config.train_batch_size}",
        f"steps={isoflop_train_config.num_train_steps}",
        f"tpu={isoflop_train_config.resources.tpu_type}",
    )
    # steps.append(
    #     default_lm_log_probs(
    #         checkpoint=isoflop_step,
    #         model=isoflop_model_config,
    #         data=eval_data,
    #         checkpoint_is_hf=False,
    #         per_device_batch_size=4,
    #         name=f"{experiment_name}-paloma-uncheatable-eval-logprobs-v2",
    #         wandb_tags=wandb_tags,
    #         resource_config=SINGLE_TPU_V5p_8_FULL,
    #     )
    # )
    # steps.append(
    #     evaluate_levanter_lm_evaluation_harness(
    #         model_name=experiment_name,
    #         model_path=isoflop_step,
    #         evals=EVAL_TASKS,
    #         resource_config=SINGLE_TPU_V5p_8_FULL,
    #         wandb_tags=wandb_tags,
    #     )
    # )

for model_config in model_with_config:
    paloma_tokenized_dict = paloma_tokenized(tokenizer=model_config.tokenizer)
    uncheatable_eval_tokenized_dict = uncheatable_eval_tokenized(tokenizer=model_config.tokenizer)
    eval_data = mixture_for_evaluation(paloma_tokenized_dict | uncheatable_eval_tokenized_dict)

    directory_friendly_name = get_directory_friendly_name(model_config.model_name)
    steps.append(
        default_lm_log_probs(
            checkpoint=model_config.model_name,
            model=model_config.model_config,
            data=eval_data,
            checkpoint_is_hf=True,
            per_device_batch_size=4,
            name=f"{directory_friendly_name}-paloma-uncheatable-eval-logprobs-v2",
            wandb_tags=[f"M={model_config.model_name}", "eval=paloma-uncheatable-eval-bpb"],
            resource_config=SINGLE_TPU_V5p_8_FULL,
        )
    )

    steps.append(
        evaluate_levanter_lm_evaluation_harness(
            model_name=f"{directory_friendly_name}-all",
            model_path=model_config.model_path,
            evals=EVAL_TASKS,
            resource_config=SINGLE_TPU_V5p_8_FULL,
            wandb_tags=[f"M={model_config.model_name}", "eval=all"],
            generation_params={
                "max_gen_toks": 1024,
                "temperature": 0.0,
                "n": 1,
                "seed": 42,
            },
        )
    )


def chunk_steps(steps: list[ExecutorStep], chunk_size: int) -> list[list[ExecutorStep]]:
    return [steps[i : i + chunk_size] for i in range(0, len(steps), chunk_size)]


if __name__ == "__main__":
    for chunk in chunk_steps(steps, 3):
        executor_main(steps=chunk)
