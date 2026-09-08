# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc evaluation definitions composed from the shared evaluation framework.

New evaluation suites use endpoint-oriented ``EvalGroup`` artifacts. The two
``evaluate_*_lm_evaluation_harness`` functions are checkpoint-native adapters
retained for historical data-mixing launchers that need direct Levanter
evaluation or custom smooth sample metrics.
"""

import logging
from collections.abc import Sequence

from fray.cluster import ResourceConfig
from marin.evaluation.evalchemy.runner import EvalchemyRunConfig
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ServeConfig
from marin.evaluation.run import evaluate
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, InputName, output_path_of, this_output_path, versioned
from marin.experiment.evaluation import EvalGroup

from experiments.evals.task_configs import (
    BASE_GENERATION_TASKS,
    CORE_TASKS,
    CORE_TASKS_PLUS_LEADERBOARD,
    KEY_GENERATION_TASKS,
    KEY_MULTIPLE_CHOICE_TASKS,
    MMLU_0_SHOT,
    MMLU_5_SHOT,
    MMLU_PRO_5_SHOT,
)

logger = logging.getLogger(__name__)

EVAL_DEPENDENCY_GROUPS = ["lm_eval", "vllm", "tpu"]
LEVANTER_EVAL_DEPENDENCY_GROUPS = ["lm_eval", "tpu"]

LM_EVAL_CODE_ENV_VARS: dict[str, str] = {
    "HF_ALLOW_CODE_EVAL": "1",
}
TPU_VLLM_ENV_VARS: dict[str, str] = {
    "MARIN_VLLM_MODE": "native",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
}


def _is_tpu_resource(resource_config: ResourceConfig | None) -> bool:
    if resource_config is None:
        return False
    return getattr(getattr(resource_config, "device", None), "kind", None) == "tpu"


def _needs_code_eval(evals: Sequence[EvalTaskConfig]) -> bool:
    return any(task.name == "humaneval" for task in evals)


def _lm_eval_env_vars(
    resource_config: ResourceConfig | None,
    evals: Sequence[EvalTaskConfig],
    env_vars: dict[str, str] | None,
) -> dict[str, str]:
    merged = dict(LM_EVAL_CODE_ENV_VARS) if _needs_code_eval(evals) else {}
    if _is_tpu_resource(resource_config):
        merged.update(TPU_VLLM_ENV_VARS)
    if env_vars:
        merged.update(env_vars)
    return merged


def evaluate_lm_evaluation_harness(
    model_name: str,
    model_path: str,
    evals: list[EvalTaskConfig],
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    generation_params: dict | None = None,
    resource_config: ResourceConfig | None = None,
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    discover_latest_checkpoint: bool = True,
    env_vars: dict[str, str] | None = None,
) -> ExecutorStep:
    """Create a legacy served LM Evaluation Harness step."""
    resources = resource_config or ResourceConfig.with_cpu()
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness/{model_name}",
        fn=remote(
            evaluate,
            resources=resources,
            env_vars=_lm_eval_env_vars(resources, evals, env_vars),
            pip_dependency_groups=EVAL_DEPENDENCY_GROUPS,
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
            generation_params=generation_params,
            resource_config=resources,
            apply_chat_template=apply_chat_template,
            wandb_tags=wandb_tags,
        ),
    )


def _infer_model_name_for_path(model_path: str) -> str:
    model_path = model_path.rstrip("/")
    return "_".join(model_path.split("/")[-2:])


def extract_model_name_and_path(step: ExecutorStep | InputName | str) -> tuple[str, InputName | str]:
    """Extract a stable model name and HF path from a legacy executor input."""
    if isinstance(step, ExecutorStep):
        return step.name, output_path_of(step, "hf")
    if isinstance(step, InputName):
        if step.step is None:
            if step.name is None:
                raise ValueError("Invalid InputName: both `step` and `name` are None.")
            return _infer_model_name_for_path(step.name), step.name
        model_path = step if step.name is not None else output_path_of(step.step, "hf")
        return step.step.name, model_path
    if isinstance(step, str):
        return _infer_model_name_for_path(step), step
    raise ValueError(f"Invalid step type: {step}")


def evaluate_levanter_lm_evaluation_harness(
    model_name: str,
    model_path: str,
    evals: list[EvalTaskConfig],
    resource_config: ResourceConfig,
    max_eval_instances: int | None = None,
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
    eval_datasets_cache_path: str | None = None,
    eval_datasets_cache_dependency: InputName | str | None = None,
    log_samples: bool = False,
    sample_log_all: bool = False,
    max_logged_samples_per_task: int | None = None,
    sample_smooth_metrics: bool = False,
    drop_samples_after_metrics: bool = False,
    use_wandb_tracker: bool = True,
) -> ExecutorStep:
    """Create a legacy direct-Levanter LM Evaluation Harness step."""
    logger.info("Running evals on the following tasks: %s", evals)
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness_levanter/lmeval_debug_{model_name}",
        fn=remote(
            evaluate,
            resources=resource_config,
            pip_dependency_groups=LEVANTER_EVAL_DEPENDENCY_GROUPS,
        ),
        config=EvaluationConfig(
            evaluator="levanter_lm_evaluation_harness",
            model_name=None,
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=versioned(evals),
            discover_latest_checkpoint=discover_latest_checkpoint,
            max_eval_instances=versioned(max_eval_instances),
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
            eval_datasets_cache_path=versioned(eval_datasets_cache_path),
            eval_datasets_cache_dependency=eval_datasets_cache_dependency,
            log_samples=log_samples,
            sample_log_all=sample_log_all,
            max_logged_samples_per_task=versioned(max_logged_samples_per_task),
            sample_smooth_metrics=sample_smooth_metrics,
            drop_samples_after_metrics=drop_samples_after_metrics,
            use_wandb_tracker=use_wandb_tracker,
        ),
    )


def _default_accelerator(accelerator: AcceleratorChoice | None) -> AcceleratorChoice:
    return accelerator or AcceleratorChoice(platform=Platform.TPU, tpu_type="v6e-8")


def core_evals(
    serve: ServeConfig | None = None,
    accelerator: AcceleratorChoice | None = None,
) -> list[EvalGroup]:
    """The core multiple-choice tasks as one served group."""
    return [
        EvalGroup(
            config=EvalchemyRunConfig(name="core", tasks=CORE_TASKS),
            serve=serve or ServeConfig(),
            accelerator=_default_accelerator(accelerator),
        )
    ]


def key_evals(
    serve: ServeConfig | None = None,
    accelerator: AcceleratorChoice | None = None,
    max_eval_instances: int | None = None,
) -> list[EvalGroup]:
    """Generation and multiple-choice key eval groups."""
    serve = serve or ServeConfig()
    accelerator = _default_accelerator(accelerator)
    return [
        EvalGroup(
            config=EvalchemyRunConfig(
                name="key_generation",
                tasks=KEY_GENERATION_TASKS,
                max_gen_toks=4096,
                max_eval_instances=max_eval_instances,
            ),
            serve=serve,
            accelerator=accelerator,
        ),
        EvalGroup(
            config=EvalchemyRunConfig(
                name="key_multiple_choice",
                tasks=KEY_MULTIPLE_CHOICE_TASKS,
                max_eval_instances=max_eval_instances,
            ),
            serve=serve,
            accelerator=accelerator,
        ),
    ]


def base_model_evals(
    serve: ServeConfig | None = None,
    accelerator: AcceleratorChoice | None = None,
    run_generation_evals: bool = True,
    discover_latest_checkpoint: bool = True,
) -> list[EvalGroup]:
    """Core, leaderboard, MMLU, and optional generation groups for base models."""
    serve = serve or ServeConfig()
    accelerator = _default_accelerator(accelerator)
    discover = discover_latest_checkpoint
    groups = [
        EvalGroup(
            EvalchemyRunConfig(name="core_leaderboard", tasks=CORE_TASKS_PLUS_LEADERBOARD),
            serve=serve,
            accelerator=accelerator,
            discover_latest_checkpoint=discover,
        ),
        EvalGroup(
            EvalchemyRunConfig(name="mmlu_0shot", tasks=(MMLU_0_SHOT,)),
            serve=serve,
            accelerator=accelerator,
            discover_latest_checkpoint=discover,
        ),
        EvalGroup(
            EvalchemyRunConfig(name="mmlu_5shot", tasks=(MMLU_5_SHOT,)),
            serve=serve,
            accelerator=accelerator,
            discover_latest_checkpoint=discover,
        ),
        EvalGroup(
            EvalchemyRunConfig(name="mmlu_pro_5shot", tasks=(MMLU_PRO_5_SHOT,)),
            serve=serve,
            accelerator=accelerator,
            discover_latest_checkpoint=discover,
        ),
    ]
    if run_generation_evals:
        groups.append(
            EvalGroup(
                EvalchemyRunConfig(
                    name="base_generation",
                    tasks=BASE_GENERATION_TASKS,
                    max_gen_toks=4096,
                ),
                serve=serve,
                accelerator=accelerator,
                discover_latest_checkpoint=discover,
            )
        )
    return groups
