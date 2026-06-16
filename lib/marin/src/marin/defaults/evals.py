# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from fray import ResourceConfig

from marin.datakit.download.uncheatable_eval import make_uncheatable_eval_step
from marin.defaults import CORE_TASKS, default_tokenize
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, InputName, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, TokenizerStep

logger = logging.getLogger(__name__)


ALL_UNCHEATABLE_EVAL_DATASETS = {
    "wikipedia_arabic": "wikipedia_arabic_*.jsonl.gz",
    "wikipedia_english": "wikipedia_english_*.jsonl.gz",
    "wikipedia_french": "wikipedia_french_*.jsonl.gz",
    "wikipedia_german": "wikipedia_german_*.jsonl.gz",
    "wikipedia_japanese": "wikipedia_japanese_*.jsonl.gz",
    "wikipedia_spanish": "wikipedia_spanish_*.jsonl.gz",
    "github_python": "github_python_*.jsonl.gz",
    "github_cpp": "github_cpp_*.jsonl.gz",
    "bbc_news": "bbc_news_*.jsonl.gz",
    "arxiv_physics": "arxiv_physics_*.jsonl.gz",
    "arxiv_computer_science": "arxiv_computer_science_*.jsonl.gz",
    "ao3_chinese": "ao3_chinese_*.jsonl.gz",
    "ao3_english": "ao3_english_*.jsonl.gz",
}
ACTIVE_DATASETS = [
    "wikipedia_english",
    "github_python",
    "github_cpp",
    "bbc_news",
    "arxiv_physics",
    "arxiv_computer_science",
    "ao3_english",
]
EVAL_DEPENDENCY_GROUPS = ["eval", "vllm", "tpu"]
EVALCHEMY_DEPENDENCY_GROUPS = ["evalchemy", "vllm", "tpu"]


DEFAULT_VLLM_ENGINE_KWARGS = {"max_model_len": 4096}
DEFAULT_LM_EVAL_MODEL_KWARGS = {**DEFAULT_VLLM_ENGINE_KWARGS, "max_gen_toks": 4096}
MMLU_0_SHOT = EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot")
MMLU_5_SHOT = EvalTaskConfig("mmlu", 5, task_alias="mmlu_5shot")
MMLU_PRO_5_SHOT = EvalTaskConfig("leaderboard_mmlu_pro", 5, task_alias="mmlu_5shot")
OPEN_LM_LEADERBOARD_MCQ = (
    EvalTaskConfig("leaderboard_bbh", 3, task_alias="lb_bbh_3shot"),
    EvalTaskConfig("leaderboard_mmlu_pro", 5, task_alias="lb_mmlu_pro_5shot"),
    EvalTaskConfig("leaderboard_gpqa", 0, task_alias="lb_gpqa_0shot"),
    EvalTaskConfig("leaderboard_musr", 0, task_alias="lb_musr_0shot"),
)
OPEN_LM_LEADERBOARD_GEN = (
    EvalTaskConfig("leaderboard_ifeval", 0, task_alias="lb_ifeval_0shot"),
    EvalTaskConfig("leaderboard_math_hard", 4, task_alias="lb_math_4shot"),
)
CORE_TASKS_PLUS_LEADERBOARD = (
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
BASE_GENERATION_TASKS = (
    EvalTaskConfig(name="bbh_cot_fewshot", num_fewshot=3),
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8),
    EvalTaskConfig(name="nq_open", num_fewshot=0, task_alias="nq_open"),
    EvalTaskConfig(name="triviaqa", num_fewshot=0, task_alias="triviaqa"),
)
KEY_GENERATION_TASKS = (
    EvalTaskConfig(name="ifeval", num_fewshot=0),
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8),
    EvalTaskConfig(name="drop", num_fewshot=0),
    EvalTaskConfig(name="humaneval", num_fewshot=10),
    EvalTaskConfig(name="bbh_cot_fewshot", num_fewshot=3, task_alias="bbh"),
    EvalTaskConfig(name="minerva_math", num_fewshot=4, task_alias="math_4shot"),
)
KEY_MULTIPLE_CHOICE_TASKS = (
    EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot"),
    EvalTaskConfig("mmlu", 5, task_alias="mmlu_5shot"),
    EvalTaskConfig(name="truthfulqa_mc2", num_fewshot=6, task_alias="truthqa"),
)


uncheatable_eval = make_uncheatable_eval_step()


def uncheatable_eval_tokenized(
    *, base_path="tokenized/", tokenizer: str | None = None, uncheatable_eval_raw: ExecutorStep = uncheatable_eval
) -> dict[str, TokenizerStep]:
    uncheatable_eval_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for dataset in ACTIVE_DATASETS:
        path_part = ALL_UNCHEATABLE_EVAL_DATASETS[dataset]
        uncheatable_eval_steps[os.path.join("uncheatable_eval", dataset)] = default_tokenize(
            name=os.path.join("uncheatable_eval", dataset),
            dataset=uncheatable_eval_raw.cd(f"{path_part}"),
            tokenizer=tokenizer,
            is_validation=True,
        )

    return uncheatable_eval_steps


def _infer_model_name_for_path(model_path: str) -> str:
    """
    Infer model name from model path.
    """
    # path names are like gs://marin-us-central2/checkpoints/dclm_7b2x/hf/dclm_7b0828/dclm_7b0828/step-479999/
    # we want something like: dclm_7b0828_step-479999
    if model_path.endswith("/"):
        model_path = model_path[:-1]

    return "_".join(model_path.split("/")[-2:])


def extract_model_name_and_path(step: ExecutorStep | InputName | str) -> tuple[str, InputName | str]:
    """
    Extract the model name and path from a step.

    Always appends /hf for ExecutorSteps; run.py's _normalize_model_path handles
    detecting whether the HF files are at root or in /hf at evaluation time.
    """
    if isinstance(step, ExecutorStep):
        model_step_path = output_path_of(step, "hf")
        name = step.name
    elif isinstance(step, InputName):
        # `InputName.hardcoded(...)` has `step.step is None`; treat it as a direct path.
        if step.step is None:
            if step.name is None:
                raise ValueError("Invalid InputName: both `step` and `name` are None.")
            model_step_path = step.name
            name = _infer_model_name_for_path(step.name)
        else:
            # If `name` is already set, the InputName refers to a specific subpath under the step's output.
            # Otherwise default to the HF export directory.
            model_step_path = step if step.name is not None else output_path_of(step.step, "hf")
            name = step.step.name
    elif isinstance(step, str):
        model_step_path = step
        name = _infer_model_name_for_path(step)
    else:
        raise ValueError(f"Invalid step type: {step}")

    return name, model_step_path


def evaluate_levanter_lm_evaluation_harness(
    model_name: str,
    model_path: str,
    evals: list[EvalTaskConfig],
    resource_config: ResourceConfig,
    max_eval_instances: int | None = None,
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using Levanter LM Evaluation Harness.
    """
    logger.info(f"Running evals on the following tasks: {evals}")
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness_levanter/lmeval_debug_{model_name}",
        fn=remote(evaluate, resources=resource_config, pip_dependency_groups=EVAL_DEPENDENCY_GROUPS),
        config=EvaluationConfig(
            evaluator="levanter_lm_evaluation_harness",
            model_name=None,  # imputed automatically
            model_path=model_path,  # type: ignore
            evaluation_path=this_output_path(),
            evals=versioned(evals),
            discover_latest_checkpoint=discover_latest_checkpoint,
            max_eval_instances=versioned(max_eval_instances),
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
        ),
    )


def evaluate_lm_evaluation_harness(
    model_name: str,
    model_path: str,
    evals: list[EvalTaskConfig],
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    resource_config: ResourceConfig | None = None,
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    discover_latest_checkpoint: bool = True,
    env_vars: dict[str, str] | None = None,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using LM Evaluation Harness.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
        evals (list[EvalTaskConfig]): List of evaluations to run with LM Evaluation Harness.
        env_vars (dict[str, str] | None): Extra env vars to set on the child iris worker.
            Needed for vLLM-on-TPU bring-up (e.g. ``VLLM_ENABLE_V1_MULTIPROCESSING=0``)
            and code-eval-dependent tasks like humaneval (``HF_ALLOW_CODE_EVAL=1``).
            The coordinator's own ``os.environ`` does NOT propagate to iris-spawned
            children — these vars must be threaded through ``remote()``.
    """
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness/{model_name}",
        fn=remote(
            evaluate,
            resources=resource_config,
            pip_dependency_groups=EVAL_DEPENDENCY_GROUPS,
            env_vars=env_vars,
        ),
        config=EvaluationConfig(
            evaluator="lm_evaluation_harness",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=evals,
            max_eval_instances=max_eval_instances,
            discover_latest_checkpoint=discover_latest_checkpoint,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
            wandb_tags=wandb_tags,
        ),
    )


def default_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v4-8"),
    evals: list[EvalTaskConfig] | None = None,
    max_eval_instances: int | None = None,
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using LM Evaluation Harness on a step.

    Args:
        step (ExecutorStep | InputName): step to evaluate.
        evals (list[EvalTaskConfig]): List of evals to run- defaults to a set of CORE_TASKS defined in task_configs.py
        max_eval_instances (int): Maximum number of evaluation instances to run.
    """

    # this logic extracts the `ExecutorStep` corresponding to the training step, and get the model path
    name, model_step_path = extract_model_name_and_path(step)

    logger.info(f"Creating default evaluation step for {name}")

    # Default to CORE_TASKS
    if evals is None:
        evals = list(CORE_TASKS)

    logger.info(f"Running evals on the following tasks: {evals}")

    return evaluate_levanter_lm_evaluation_harness(
        name,
        model_step_path,
        evals,
        resource_config,
        max_eval_instances=max_eval_instances,
        apply_chat_template=apply_chat_template,
        discover_latest_checkpoint=discover_latest_checkpoint,
    )


def default_base_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v6e-8"),
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_LM_EVAL_MODEL_KWARGS,
    run_generation_evals: bool = True,
    discover_latest_checkpoint: bool = True,
):
    # Add GPQA to CORE_TASKS
    # Set up evaluations for core tasks (including GPQA)
    eval_jobs = []
    core_grouped = default_eval(
        step=step,
        resource_config=resource_config,
        evals=CORE_TASKS_PLUS_LEADERBOARD,
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    eval_jobs.append(core_grouped)

    # Run tasks where we report Macro_Avg separately to make sure the macro avg gets computed correctly.
    mmlu_0shot = default_eval(
        step=step,
        resource_config=resource_config,
        evals=(MMLU_0_SHOT,),
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    eval_jobs.append(mmlu_0shot)

    mmlu_5shot = default_eval(
        step=step,
        resource_config=resource_config,
        evals=(MMLU_5_SHOT,),
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    eval_jobs.append(mmlu_5shot)

    mmlu_pro_5shot = default_eval(
        step=step,
        resource_config=resource_config,
        evals=(MMLU_PRO_5_SHOT,),
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    eval_jobs.append(mmlu_pro_5shot)

    name, model_step_path = extract_model_name_and_path(step)
    if run_generation_evals:
        generation = evaluate_lm_evaluation_harness(
            name,
            model_step_path,
            BASE_GENERATION_TASKS,
            max_eval_instances=max_eval_instances,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
            discover_latest_checkpoint=discover_latest_checkpoint,
        )

        eval_jobs.append(generation)
    return eval_jobs


def default_sft_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v6e-8"),
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_LM_EVAL_MODEL_KWARGS,
    run_generation_evals: bool = True,
    apply_chat_template: bool = True,
    use_levanter_inference: bool = False,
):
    # Set up evaluations for core tasks (including GPQA)
    eval_jobs = []
    leaderboard_grouped = default_eval(
        step=step,
        resource_config=resource_config,
        evals=OPEN_LM_LEADERBOARD_MCQ,
        apply_chat_template=apply_chat_template,
    )
    eval_jobs.append(leaderboard_grouped)

    # Run tasks where we report Macro_Avg separately to make sure the macro avg gets computed correctly.

    mmlu_5shot = default_eval(
        step=step, resource_config=resource_config, evals=(MMLU_5_SHOT,), apply_chat_template=apply_chat_template
    )
    eval_jobs.append(mmlu_5shot)

    mmlu_pro_5shot = default_eval(
        step=step, resource_config=resource_config, evals=(MMLU_PRO_5_SHOT,), apply_chat_template=apply_chat_template
    )
    eval_jobs.append(mmlu_pro_5shot)

    name, model_step_path = extract_model_name_and_path(step)
    if run_generation_evals:
        if use_levanter_inference:
            leaderboard_generation = evaluate_levanter_lm_evaluation_harness(
                name,
                model_step_path,
                KEY_GENERATION_TASKS,
                resource_config,
                max_eval_instances=max_eval_instances,
                apply_chat_template=apply_chat_template,
            )
            eval_jobs.append(leaderboard_generation)

            olmo_generation = evaluate_levanter_lm_evaluation_harness(
                name,
                model_step_path,
                OPEN_LM_LEADERBOARD_GEN,
                resource_config,
                max_eval_instances=max_eval_instances,
                apply_chat_template=apply_chat_template,
            )
            eval_jobs.append(olmo_generation)
        else:
            leaderboard_generation = evaluate_lm_evaluation_harness(
                name,
                model_step_path,
                KEY_GENERATION_TASKS,
                max_eval_instances=max_eval_instances,
                engine_kwargs=engine_kwargs,
                resource_config=resource_config,
                apply_chat_template=apply_chat_template,
            )

            eval_jobs.append(leaderboard_generation)

            olmo_generation = evaluate_lm_evaluation_harness(
                name,
                model_step_path,
                OPEN_LM_LEADERBOARD_GEN,
                max_eval_instances=max_eval_instances,
                engine_kwargs=engine_kwargs,
                resource_config=resource_config,
                apply_chat_template=apply_chat_template,
            )
            eval_jobs.append(olmo_generation)
    return eval_jobs


def default_key_evals(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig,
    model_name: str | None = None,
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_LM_EVAL_MODEL_KWARGS,
) -> list[ExecutorStep]:
    """
    Create a list of ExecutorSteps to evaluate the model using LM Evaluation Harness on a step.
    """
    name, model_step_path = extract_model_name_and_path(step)

    if model_name is None:
        model_name = name

    stop_token_ids = []
    if "llama3" in model_name:
        stop_token_ids.append(128009)
    elif "olmo" in model_name:
        stop_token_ids.append(100257)

    return [
        evaluate_lm_evaluation_harness(
            model_name,
            model_step_path,
            KEY_GENERATION_TASKS,
            max_eval_instances=max_eval_instances,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
        ),
        evaluate_levanter_lm_evaluation_harness(
            model_name,
            model_step_path,
            KEY_MULTIPLE_CHOICE_TASKS,
            resource_config,
            max_eval_instances=max_eval_instances,
        ),
    ]
