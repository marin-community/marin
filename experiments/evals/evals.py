# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint evaluation menus using exact curated protocol variants."""

from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ServeConfig

from experiments.evals.evaluation import EvalGroup

CORE = (
    "agieval-lsat-ar-3shot",
    "arc-easy-10shot",
    "arc-challenge-10shot",
    "boolq-10shot",
    "commonsense-qa-10shot",
    "copa",
    "hellaswag-0shot",
    "hellaswag",
    "lambada",
    "openbookqa",
    "piqa-10shot",
    "wsc273",
    "winogrande-0shot",
)


def core_evals(serve: ServeConfig | None = None, accelerator: AcceleratorChoice | None = None) -> list[EvalGroup]:
    return [
        EvalGroup(
            name="core",
            evals=CORE,
            serve=serve or ServeConfig(),
            accelerator=accelerator or AcceleratorChoice(platform=Platform.TPU, tpu_type="v6e-8"),
        )
    ]


def key_evals(
    serve: ServeConfig | None = None, accelerator: AcceleratorChoice | None = None, max_eval_instances: int | None = None
) -> list[EvalGroup]:
    serving = serve or ServeConfig()
    hardware = accelerator or AcceleratorChoice(platform=Platform.TPU, tpu_type="v6e-8")
    return [
        EvalGroup(
            name="key_generation",
            evals=("ifeval", "gsm8k-cot-8shot", "drop-0shot", "humaneval-10shot", "bbh-cot-3shot", "minerva-math-4shot"),
            serve=serving,
            accelerator=hardware,
            max_tokens=4096,
            limit=max_eval_instances,
        ),
        EvalGroup(
            name="key_multiple_choice",
            evals=("mmlu-0shot", "mmlu", "truthfulqa-6shot"),
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
    hardware = accelerator or AcceleratorChoice(platform=Platform.TPU, tpu_type="v6e-8")
    selections = [
        ("core_leaderboard", ("bbh-3shot", "gpqa-0shot", *CORE)),
        ("mmlu_0shot", ("mmlu-0shot",)),
        ("mmlu_5shot", ("mmlu",)),
        ("mmlu_pro_5shot", ("mmlu-pro-5shot",)),
    ]
    if run_generation_evals:
        selections.append(("base_generation", ("bbh-cot-3shot", "gsm8k-cot-8shot", "nq-open-0shot", "triviaqa-0shot")))
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
