"""
Base model evaluations across multiple LLMs.

This experiment evaluates OLMO Base 8B, LLAMA 3.1 8B, Deeper Starling 8B,
MAP-NEO 7B, and Amber Base 7B models on CORE_TASKS (augmented with OLMo Eval Tasks) as well
as dedicated MMLU 0-shot and 5-shot configurations.
"""

from experiments.evals.engine_configs import DEFAULT_VLLM_ENGINE_KWARGS
from experiments.evals.evals import default_eval, evaluate_lm_evaluation_harness, extract_model_name_and_path
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8, ResourceConfig
from experiments.evals.task_configs import CORE_TASKS
from experiments.models import amber_base_7b, llama_3_1_8b, map_neo_7b, olmo_2_base_8b
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main


def run_core_base_evals(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = SINGLE_TPU_V6E_8,
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_VLLM_ENGINE_KWARGS,
    run_generation_evals: bool = True,
):
    # Add GPQA to CORE_TASKS

    core_tasks_with_leaderboard = (
        EvalTaskConfig(
            "leaderboard_bbh",
            3,
            task_alias="bbh_3shot",
        ),
        EvalTaskConfig(
            "leaderboard_gpqa",
            0,
            task_alias="gpqa_0shot",
        ),
        *CORE_TASKS,
    )
    # Create EvalTaskConfig objects for tasks that need to be run on their own for Macro Avg purposes in LM Eval harness
    mmlu_0shot_config = [EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot")]
    mmlu_5shot_config = [EvalTaskConfig("mmlu", 5, task_alias="mmlu_5shot")]
    mmlu_pro_5shot_config = [EvalTaskConfig("leaderboard_mmlu_pro", 5, task_alias="mmlu_5shot")]

    generation_tasks = (
        EvalTaskConfig(name="gsm8k_cot", num_fewshot=8),
        EvalTaskConfig(name="nq_open", num_fewshot=0, task_alias="nq_open"),
        EvalTaskConfig(name="triviaqa", num_fewshot=0, task_alias="triviaqa"),
    )

    # Set up evaluations for core tasks (including GPQA)
    eval_jobs = []
    core_grouped = default_eval(
        step=step,
        resource_config=resource_config,
        evals=core_tasks_with_leaderboard,
    )
    eval_jobs.append(core_grouped)

    mmlu_0shot = default_eval(
        step=step,
        resource_config=resource_config,
        evals=mmlu_0shot_config,
    )
    eval_jobs.append(mmlu_0shot)

    mmlu_5shot = default_eval(
        step=step,
        resource_config=resource_config,
        evals=mmlu_5shot_config,
    )
    eval_jobs.append(mmlu_5shot)

    mmlu_pro_5shot = default_eval(
        step=step,
        resource_config=resource_config,
        evals=mmlu_pro_5shot_config,
    )
    eval_jobs.append(mmlu_pro_5shot)

    name, model_step_path = extract_model_name_and_path(step)
    generation = evaluate_lm_evaluation_harness(
        name + "_mmlu_pro",
        model_step_path,
        generation_tasks,
        max_eval_instances=max_eval_instances,
        engine_kwargs=engine_kwargs,
        resource_config=resource_config,
    )

    eval_jobs.append(generation)
    return eval_jobs


if __name__ == "__main__":
    # Model path for deeper starling
    deeper_starling_path = "gs://marin-us-central2/checkpoints/tootsie-8b-deeper-starling/hf/step-1419999"
    # Run all evaluations on all models
    executor_main(
        steps=[
            *run_core_base_evals(deeper_starling_path),
            *run_core_base_evals(llama_3_1_8b),
            *run_core_base_evals(olmo_2_base_8b),
            *run_core_base_evals(amber_base_7b, engine_kwargs={"max_model_len": 2048, "max_gen_toks": 2048}),
            *run_core_base_evals(
                map_neo_7b, engine_kwargs={"trust_remote_code": True, "max_model_len": 4096, "max_gen_toks": 4096}
            ),
        ]
    )
