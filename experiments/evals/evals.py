# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint evaluation menus using exact curated protocol variants."""

from evalrecords.records import EvaluationSelection
from marin.evaluation.hardware import AcceleratorChoice
from marin.evaluation.model_config import ServeConfig

from experiments.evals.evaluation import DEFAULT_ACCELERATOR, EvalGroup

CORE = (
    EvaluationSelection(benchmark="agieval-lsat-ar", protocol="likelihood", num_fewshot=3),
    EvaluationSelection(benchmark="arc-easy", protocol="harness", num_fewshot=10),
    EvaluationSelection(benchmark="arc-challenge", protocol="harness", num_fewshot=10),
    EvaluationSelection(benchmark="boolq", protocol="harness", num_fewshot=10),
    EvaluationSelection(benchmark="commonsense-qa", protocol="likelihood", num_fewshot=10),
    EvaluationSelection(benchmark="copa", protocol="likelihood", num_fewshot=0),
    EvaluationSelection(benchmark="hellaswag", protocol="harness", num_fewshot=0),
    EvaluationSelection(benchmark="hellaswag", protocol="harness", num_fewshot=10),
    EvaluationSelection(benchmark="lambada", protocol="harness"),
    EvaluationSelection(benchmark="openbookqa", protocol="harness", num_fewshot=0),
    EvaluationSelection(benchmark="piqa", protocol="harness", num_fewshot=10),
    EvaluationSelection(benchmark="wsc273", protocol="likelihood", num_fewshot=0),
    EvaluationSelection(benchmark="winogrande", protocol="harness", num_fewshot=0),
)


def core_evals(serve: ServeConfig | None = None, accelerator: AcceleratorChoice | None = None) -> list[EvalGroup]:
    return [
        EvalGroup(
            name="core",
            evals=CORE,
            serve=serve or ServeConfig(),
            accelerator=accelerator or DEFAULT_ACCELERATOR,
        )
    ]


def key_evals(
    serve: ServeConfig | None = None, accelerator: AcceleratorChoice | None = None, max_eval_instances: int | None = None
) -> list[EvalGroup]:
    serving = serve or ServeConfig()
    hardware = accelerator or DEFAULT_ACCELERATOR
    return [
        EvalGroup(
            name="key_generation",
            evals=(
                EvaluationSelection(benchmark="ifeval", protocol="evalchemy"),
                EvaluationSelection(benchmark="gsm8k", protocol="cot", num_fewshot=8),
                EvaluationSelection(benchmark="drop", protocol="harness", num_fewshot=0),
                EvaluationSelection(benchmark="humaneval", protocol="evalchemy", num_fewshot=10),
                EvaluationSelection(benchmark="bbh", protocol="cot", num_fewshot=3),
                EvaluationSelection(benchmark="hendrycks-math", protocol="cot", num_fewshot=4),
            ),
            serve=serving,
            accelerator=hardware,
            max_tokens=4096,
            limit=max_eval_instances,
        ),
        EvalGroup(
            name="key_multiple_choice",
            evals=(
                EvaluationSelection(benchmark="mmlu", protocol="harness", num_fewshot=0),
                EvaluationSelection(benchmark="mmlu", protocol="harness", num_fewshot=5),
                EvaluationSelection(benchmark="truthfulqa", protocol="harness", num_fewshot=6),
            ),
            serve=serving,
            accelerator=hardware,
            limit=max_eval_instances,
        ),
    ]


def base_model_evals(
    serve: ServeConfig | None = None,
    accelerator: AcceleratorChoice | None = None,
    run_generation_evals: bool = True,
    discover_latest_checkpoint: bool = True,
) -> list[EvalGroup]:
    serving = serve or ServeConfig()
    hardware = accelerator or DEFAULT_ACCELERATOR
    selections = [
        (
            "core_leaderboard",
            (
                EvaluationSelection(benchmark="bbh", protocol="likelihood", num_fewshot=3),
                EvaluationSelection(benchmark="gpqa", protocol="likelihood", num_fewshot=0),
                *CORE,
            ),
        ),
        ("mmlu_0shot", (EvaluationSelection(benchmark="mmlu", protocol="harness", num_fewshot=0),)),
        ("mmlu_5shot", (EvaluationSelection(benchmark="mmlu", protocol="harness", num_fewshot=5),)),
        ("mmlu_pro_5shot", (EvaluationSelection(benchmark="mmlu-pro", protocol="likelihood", num_fewshot=5),)),
    ]
    if run_generation_evals:
        selections.append(
            (
                "base_generation",
                (
                    EvaluationSelection(benchmark="bbh", protocol="cot", num_fewshot=3),
                    EvaluationSelection(benchmark="gsm8k", protocol="cot", num_fewshot=8),
                    EvaluationSelection(benchmark="nq-open", protocol="harness", num_fewshot=0),
                    EvaluationSelection(benchmark="triviaqa", protocol="harness", num_fewshot=0),
                ),
            )
        )
    return [
        EvalGroup(
            name=name,
            evals=evals,
            serve=serving,
            accelerator=hardware,
            discover_latest_checkpoint=discover_latest_checkpoint,
            max_tokens=4096 if name == "base_generation" else None,
        )
        for name, evals in selections
    ]
