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
