# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc evaluation definitions composed from the shared evaluation framework."""

from marin.evaluation.evalchemy import EvalchemyRunConfig
from marin.evaluation.serving_config import ServeSpec
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


def core_evals(serve: ServeSpec | None = None) -> list[EvalGroup]:
    """The core multiple-choice tasks as one served group."""
    serve = serve or ServeSpec(tpu_type="v6e-8")
    return [EvalGroup(config=EvalchemyRunConfig(name="core", tasks=CORE_TASKS), serve=serve)]


def key_evals(
    serve: ServeSpec | None = None,
    max_eval_instances: int | None = None,
) -> list[EvalGroup]:
    """Generation and multiple-choice key eval groups."""
    serve = serve or ServeSpec(tpu_type="v6e-8")
    return [
        EvalGroup(
            config=EvalchemyRunConfig(
                name="key_generation",
                tasks=KEY_GENERATION_TASKS,
                max_gen_toks=4096,
                max_eval_instances=max_eval_instances,
            ),
            serve=serve,
        ),
        EvalGroup(
            config=EvalchemyRunConfig(
                name="key_multiple_choice",
                tasks=KEY_MULTIPLE_CHOICE_TASKS,
                max_eval_instances=max_eval_instances,
            ),
            serve=serve,
        ),
    ]


def base_model_evals(
    serve: ServeSpec | None = None,
    run_generation_evals: bool = True,
    discover_latest_checkpoint: bool = True,
) -> list[EvalGroup]:
    """Core, leaderboard, MMLU, and optional generation groups for base models."""
    serve = serve or ServeSpec(tpu_type="v6e-8")
    discover = discover_latest_checkpoint
    groups = [
        EvalGroup(
            EvalchemyRunConfig(name="core_leaderboard", tasks=CORE_TASKS_PLUS_LEADERBOARD),
            serve=serve,
            discover_latest_checkpoint=discover,
        ),
        EvalGroup(
            EvalchemyRunConfig(name="mmlu_0shot", tasks=(MMLU_0_SHOT,)),
            serve=serve,
            discover_latest_checkpoint=discover,
        ),
        EvalGroup(
            EvalchemyRunConfig(name="mmlu_5shot", tasks=(MMLU_5_SHOT,)),
            serve=serve,
            discover_latest_checkpoint=discover,
        ),
        EvalGroup(
            EvalchemyRunConfig(name="mmlu_pro_5shot", tasks=(MMLU_PRO_5_SHOT,)),
            serve=serve,
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
                discover_latest_checkpoint=discover,
            )
        )
    return groups
