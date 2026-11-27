# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind task templates through explicitly supplied machine factories."""

import json
from collections.abc import Callable, Mapping
from contextlib import ExitStack

from rolloutengine.machines import machine_spec
from rolloutengine.models import MachineRuntimeSpec
from rolloutengine.validation import task_machine_reason
from shellbox.context import MachineRoute, ShellboxContext
from shellbox.machine import Backend, MachineFactory, UnsupportedMachineSpec
from skyrlrecipe.declared import TASK_MACHINE_BY_TIER
from taskcompendium.models import EnvironmentRequirements, TaskSpec, grader_environment, grades_with_judge
from taskcompendium.runtime.requirements import ROUTE_BACKENDS_LABEL


def tier_machine(
    requirements: EnvironmentRequirements,
    machines_by_tier: Mapping[str, dict],
    recorded: tuple[Backend, ...],
    factory_provider: Callable[[], Mapping[str, MachineFactory]],
) -> dict | None:
    """Select the first configured factory that admits the task and allocation."""
    if requirements.command_semantics is None or not machines_by_tier:
        return None
    try:
        normalized = requirements.requirements
    except UnsupportedMachineSpec:
        return None
    factories = factory_provider()
    candidates = []
    with ExitStack() as stack:
        for machine in machines_by_tier.values():
            runtime = MachineRuntimeSpec.model_validate(machine)
            factory = factories.get(runtime.backend)
            if factory is None or (recorded and factory.backend not in recorded):
                continue
            try:
                spec = stack.enter_context(machine_spec(requirements, runtime))
            except UnsupportedMachineSpec:
                return None
            candidates.append((MachineRoute(factory, spec), machine))
        admitted = ShellboxContext(tuple(route for route, _ in candidates)).routes(normalized)
        if not admitted:
            return None
        return next(machine for route, machine in candidates if route is admitted[0])


def task_machine(
    task: TaskSpec, config: Mapping[str, dict], factory_provider: Callable[[], Mapping[str, MachineFactory]]
) -> MachineRuntimeSpec | None:
    """Bind a task runtime while retaining its verified backend restrictions."""
    if task_machine_reason(task) is None:
        return None
    machine = config["task_machine_by_dataset"].get(task.source.dataset)
    recorded = task.labels.get(ROUTE_BACKENDS_LABEL)
    if machine is None or recorded is not None:
        templates = config.get(TASK_MACHINE_BY_TIER, {}) if machine is None else {machine["backend"]: machine}
        machine = tier_machine(
            task.environment_requirements,
            templates,
            () if recorded is None else tuple(Backend(value) for value in json.loads(recorded)),
            factory_provider,
        )
    return MachineRuntimeSpec.model_validate(machine) if machine is not None else None


def verifier_machine(task: TaskSpec, config: Mapping[str, dict]) -> MachineRuntimeSpec | None:
    """Bind the verifier runtime for machine-backed graders."""
    environment = grader_environment(task.grader)
    if environment is None or grades_with_judge(task.grader):
        return None
    machine = config["image_verifier"] if environment.docker_image is not None else config["package_verifier"]
    return MachineRuntimeSpec.model_validate(machine)
