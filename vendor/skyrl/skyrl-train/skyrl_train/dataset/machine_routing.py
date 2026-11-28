# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind task templates through explicitly supplied machine factories.

Every stage selects machines the same way: its configured factories in deployment
order, admitted per actual task environment, with the task's verification evidence
excluding any backend whose control failed or was rejected for that role. A task
without evidence has no exclusions; an explicit per-dataset binding stays trusted.
"""

from collections.abc import Callable, Mapping
from contextlib import ExitStack

from rolloutengine.machines import machine_spec
from rolloutengine.models import MachineRuntimeSpec
from rolloutengine.validation import task_machine_reason
from shellbox.context import MachineRoute, ShellboxContext
from shellbox.machine import MachineFactory, UnsupportedMachineSpec
from skyrlrecipe.declared import TASK_MACHINES_BY_BACKEND
from taskcompendium.models import EnvironmentRequirements, TaskSpec, grader_environment, grades_with_judge
from taskcompendium.pipeline.machine_routes import excluded_backends, task_route_evidence
from taskcompendium.pipeline.models import MachineRole


def _excluded_backends(task: TaskSpec, role: MachineRole) -> frozenset[str]:
    """Backends whose recorded control outcome for the role excludes selection."""
    return excluded_backends(task_route_evidence(task), role)


def backend_machine(
    requirements: EnvironmentRequirements,
    machines_by_backend: Mapping[str, dict],
    excluded: frozenset[str],
    factory_provider: Callable[[], Mapping[str, MachineFactory]],
) -> dict | None:
    """Select the first configured factory that admits the task, allocation and evidence."""
    if requirements.command_semantics is None or not machines_by_backend:
        return None
    try:
        normalized = requirements.requirements
    except UnsupportedMachineSpec:
        return None
    factories = factory_provider()
    candidates = []
    with ExitStack() as stack:
        for machine in machines_by_backend.values():
            runtime = MachineRuntimeSpec.model_validate(machine)
            factory = factories.get(runtime.backend)
            if factory is None or runtime.backend in excluded:
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
    """Bind a task runtime that its verified evidence does not exclude."""
    if task_machine_reason(task) is None:
        return None
    evidence = task_route_evidence(task)
    excluded = excluded_backends(evidence, MachineRole.TASK)
    machine = config["task_machine_by_dataset"].get(task.source.dataset)
    if machine is None or evidence:
        templates = config.get(TASK_MACHINES_BY_BACKEND, {}) if machine is None else {machine["backend"]: machine}
        machine = backend_machine(
            task.environment_requirements,
            templates,
            excluded,
            factory_provider,
        )
    return MachineRuntimeSpec.model_validate(machine) if machine is not None else None


def verifier_machine(task: TaskSpec, config: Mapping[str, dict]) -> MachineRuntimeSpec | None:
    """Bind the verifier runtime for machine-backed graders the evidence does not exclude."""
    environment = grader_environment(task.grader)
    if environment is None or grades_with_judge(task.grader):
        return None
    machine = config["image_verifier"] if environment.docker_image is not None else config["package_verifier"]
    runtime = MachineRuntimeSpec.model_validate(machine)
    if runtime.backend in _excluded_backends(task, MachineRole.VERIFIER):
        return None
    return runtime
