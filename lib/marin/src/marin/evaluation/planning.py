# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure planning over selected evaluation definitions."""

from dataclasses import dataclass

from marin.evaluation.definitions import EvalDefinition
from marin.evaluation.hardware import (
    AcceleratorChoice,
    HardwarePolicy,
    Platform,
    select_accelerator,
)
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.serving_config import ServeSpec, serve_spec_for_model


@dataclass(frozen=True)
class EvalLaunchRequest:
    """Selected experiment values supplied to the generic planner."""

    model_key: str
    model: ModelConfig
    evaluations: tuple[tuple[str, EvalDefinition], ...]
    platform: Platform
    accelerator: str | None
    limit: int | None


@dataclass(frozen=True)
class RunPlan:
    """One model/evaluation pair resolved to hardware and serving policy."""

    model_key: str
    eval_key: str
    model: ModelConfig
    suite: EvalDefinition
    accel: AcceleratorChoice
    serve: ServeSpec
    limit: int | None


def plan_runs(
    request: EvalLaunchRequest,
    hardware: HardwarePolicy,
) -> list[RunPlan]:
    """Resolve a selected request without importing experiment catalogs."""
    if not request.evaluations:
        raise ValueError("at least one evaluation is required")
    mechanisms = {type(definition) for _, definition in request.evaluations}
    if len(mechanisms) != 1:
        raise ValueError("a launch group cannot mix Evalchemy and Harbor definitions")
    accelerator = select_accelerator(
        request.model,
        request.platform,
        request.accelerator,
        hardware,
    )
    serve = serve_spec_for_model(request.model, accelerator)
    return [
        RunPlan(
            model_key=request.model_key,
            eval_key=eval_key,
            model=request.model,
            suite=definition,
            accel=accelerator,
            serve=serve,
            limit=request.limit if request.limit is not None else definition.max_eval_instances,
        )
        for eval_key, definition in request.evaluations
    ]
