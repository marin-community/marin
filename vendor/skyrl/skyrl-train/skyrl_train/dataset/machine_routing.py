# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind task templates through the rollout workers' configured factories."""

import json
import os
from collections.abc import Mapping
from dataclasses import replace
from functools import cache

from marinskyrl.task_machine_factories import task_machine_factories
from omegaconf import OmegaConf
from rolloutengine.models import MachineRuntimeSpec
from shellbox.context import MachineRoute, ShellboxContext, source_spec
from shellbox.machine import Backend, MachineCapability, MachineFactory, NetworkPolicy, UnsupportedMachineSpec
from taskcompendium.models import EnvironmentRequirements


@cache
def _configured_factories(configuration: str, process_id: int) -> Mapping[str, MachineFactory]:
    # The PID in the cache key prevents forked mappers from reusing a parent's
    # native simulator state; each process constructs its own factories once.
    assert process_id == os.getpid()
    return task_machine_factories(OmegaConf.create(json.loads(configuration)))


def tier_machine(
    requirements: EnvironmentRequirements,
    machines_by_tier: Mapping[str, dict],
    recorded: tuple[Backend, ...],
    environment_configs: Mapping[str, dict],
    factories: Mapping[str, MachineFactory] | None,
) -> dict | None:
    """Select the first configured factory that admits the task and allocation."""
    if requirements.command_semantics is None or not machines_by_tier:
        return None
    try:
        normalized = requirements.requirements
    except UnsupportedMachineSpec:
        return None
    if factories is None:
        factories = _configured_factories(json.dumps(environment_configs, sort_keys=True), os.getpid())
    candidates = []
    for machine in machines_by_tier.values():
        runtime = MachineRuntimeSpec.model_validate(machine)
        factory = factories.get(runtime.backend)
        if factory is None or (recorded and factory.backend not in recorded):
            continue
        network = NetworkPolicy.ALLOW if MachineCapability.NETWORK in normalized.capabilities else runtime.network
        try:
            spec = replace(
                source_spec(requirements, network),
                cpus=runtime.cpus,
                memory_mb=runtime.memory_mb,
                storage_mb=runtime.storage_mb,
                gpus=runtime.gpus,
                startup_timeout=runtime.startup_timeout,
            )
        except UnsupportedMachineSpec:
            return None
        candidates.append((MachineRoute(factory, spec), {**machine, "network": network.value}))
    admitted = ShellboxContext(tuple(route for route, _ in candidates)).routes(normalized)
    if not admitted:
        return None
    return next(machine for route, machine in candidates if route is admitted[0])
