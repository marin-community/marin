# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-running launcher for experiment scripts.

Run an experiment script directly, choosing where it executes::

    uv run python experiments/grug/base/launch.py --cluster=marin   # on the cluster
    uv run python experiments/grug/base/launch.py --local           # in this process

:func:`launch` runs the script's body in-process (``--local``, a dry run, or no
``--cluster``), otherwise it ships the body to a small CPU *coordinator* job that
drives the executor DAG (or direct submits) and spawns training as its children.
The coordinator outlives this process, so closing the laptop never strands a run
— its ``.success`` markers are written on the cluster. :func:`launch_executor` is
the executor-step form of :func:`launch`; :func:`override_resources` applies
``--region`` / ``--tpu_type`` to a script's :class:`ResourceConfig`.
"""

from __future__ import annotations

import dataclasses
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fray.types import GpuConfig, ResourceConfig, TpuConfig, get_tpu_topology
from iris.cli.job import add_standard_env_vars, load_env_vars
from iris.client.connect import connect_to_cluster, stream_until_complete
from iris.cluster.client.job_info import get_job_info
from iris.cluster.constraints import Constraint, region_constraint, zone_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main
from marin.execution.types import VersionedValue, versioned

logger = logging.getLogger(__name__)

# The coordinator is a small CPU job that runs the executor/DAG (or direct
# submit) body on the cluster, so the run survives the launching laptop
# disconnecting. It only schedules and dispatches work — CPU is enough, even
# though the DAG runner imports levanter/marin and holds a thread pool.
COORDINATOR_CPU = 1.0
COORDINATOR_MEMORY = "4GB"
COORDINATOR_DISK = "10GB"


@dataclass
class LaunchConfig:
    """CLI surface for a self-running experiment script.

    Parsed with draccus (``--cluster=marin --tpu_type=v5p-8 --region=...``).
    Embeds :class:`ExecutorMainConfig` so executor flags remain available as
    ``--executor.dry_run`` etc.
    """

    cluster: str | None = None
    """Named Iris cluster to connect to (e.g. ``marin``). When unset, the script
    runs against the in-process LocalClient — no cluster connection."""

    region: str | None = None
    """Restrict submitted jobs to this region (overrides the script's resources)."""

    zone: str | None = None
    """Restrict submitted jobs to this zone (overrides the script's resources)."""

    tpu_type: str | None = None
    """Override the script's TPU type (e.g. ``v4-8``). Must share vm_count with the
    script default; multi-slice / flexible configs must be edited in the script."""

    local: bool = False
    """Force local in-process execution even when ``--cluster`` is given."""

    detach: bool = False
    """Submit the coordinator job and return immediately instead of streaming
    its logs. The coordinator — and the training it spawns — keeps running on
    the cluster; reconnect with ``iris job logs -f <id>``. Default streams and
    blocks until the run finishes."""

    executor: ExecutorMainConfig = field(default_factory=ExecutorMainConfig)
    """Embedded executor flags (``--executor.dry_run``, ``--executor.prefix``, ...)."""


def _repo_root() -> Path:
    """Return the git workspace root to bundle for workers.

    This file lives at ``<root>/experiments/launch.py``, so the root is two
    parents up; we assert ``pyproject.toml`` is there so the bundle is rooted at
    a real workspace rather than some unexpected cwd.
    """
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"Expected marin repo root with pyproject.toml at {root}, found none.")
    return root


def launch(config: LaunchConfig, body: Callable[..., None], *args, **kwargs) -> None:
    """Run a script's ``body`` in-process, or as a coordinator job on the cluster.

    ``body`` is the script's actual work — ``train`` / ``train_grug`` /
    ``executor_main`` / a submit loop — invoked as ``body(*args, **kwargs)``.

    It runs in-process when staying local (``--local``, a dry run, or no
    ``--cluster``) or when already inside the coordinator we submitted. Otherwise
    it ships ``body`` to a small CPU *coordinator* job that runs it on the cluster
    and spawns training as its children, then streams that job's logs and exits
    with its status (``--detach`` returns right after submit). Because ``body``
    runs on the cluster, the executor DAG and its ``.success`` markers survive the
    laptop disconnecting.

    To be shipped, ``body`` must be importable on the worker (a top-level
    function) and its ``args``/``kwargs`` cloudpickle-able.
    """
    if get_job_info() is not None or config.local or config.cluster is None or config.executor.dry_run:
        body(*args, **kwargs)
        return
    raise SystemExit(_submit_coordinator_job(config, body, args, kwargs))


def _coordinator_job_name(body: Callable[..., None]) -> str:
    """A collision-free coordinator job name derived from the body callable."""
    module = body.__module__.rsplit(".", 1)[-1]
    return f"{module}-{body.__name__}-{uuid.uuid4().hex[:8]}"


def _coordinator_constraints(config: LaunchConfig) -> list[Constraint] | None:
    """Pin the coordinator to ``--region`` / ``--zone`` when one is given.

    The coordinator resolves ``marin_prefix()`` in its own region and bakes every
    executor output path from it, so on the executor path it must land in the same
    region the run's accelerators do (``--region`` already constrains the training
    steps via :func:`override_resources`). When unset, the scheduler places the
    coordinator and the executor's per-step region inference keeps each accelerator
    with the bucket it baked.
    """
    constraints: list[Constraint] = []
    if config.region is not None:
        constraints.append(region_constraint([config.region]))
    if config.zone is not None:
        constraints.append(zone_constraint(config.zone))
    return constraints or None


def _submit_coordinator_job(config: LaunchConfig, body: Callable[..., None], args: tuple, kwargs: dict) -> int:
    """Submit a CPU coordinator job that runs ``body`` on the cluster.

    ``body`` and its args are cloudpickled via :meth:`Entrypoint.from_callable`,
    so the coordinator runs the executor DAG (or direct submits) in-cluster and
    spawns training as its children while this process only streams its logs.
    Returns the coordinator's exit code.
    """
    assert config.cluster is not None
    env_vars = add_standard_env_vars(load_env_vars(None))
    resources = ResourceSpec(cpu=COORDINATOR_CPU, memory=COORDINATOR_MEMORY, disk=COORDINATOR_DISK)
    logger.info("Launching coordinator on cluster %r for %s", config.cluster, body.__qualname__)
    with connect_to_cluster(config.cluster, workspace=_repo_root()) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_callable(body, *args, **kwargs),
            name=_coordinator_job_name(body),
            resources=resources,
            environment=EnvironmentSpec(env_vars=env_vars, extras=["cpu"]),
            constraints=_coordinator_constraints(config),
        )
        if config.detach:
            logger.info("Detached; the coordinator keeps running. Reconnect with: iris job logs -f %s", job.job_id)
            return 0
        return stream_until_complete(client, job)


def override_resources(resources: ResourceConfig, config: LaunchConfig) -> ResourceConfig:
    """Apply ``--tpu_type`` / ``--region`` / ``--zone`` to a script's resources.

    Preserves every other scheduling field (replicas, preemptible, image, ...).
    Rejects unsafe TPU swaps — a different ``vm_count`` would silently corrupt the
    replica count, and a flexible (multi-variant) config can't be re-derived from
    a single ``--tpu_type``. Edit those in the script instead.
    """
    if config.tpu_type is not None:
        if not isinstance(resources.device, TpuConfig):
            raise ValueError(f"--tpu_type only applies to TPU resources, got {type(resources.device).__name__}.")
        if resources.device_alternatives:
            raise ValueError("--tpu_type cannot override a flexible multi-variant TPU config; edit the script.")
        current_vm_count = get_tpu_topology(resources.device.variant).vm_count
        new_vm_count = get_tpu_topology(config.tpu_type).vm_count
        if new_vm_count != current_vm_count:
            raise ValueError(
                f"--tpu_type={config.tpu_type} (vm_count={new_vm_count}) differs in vm_count from the "
                f"script default {resources.device.variant} (vm_count={current_vm_count}); the replica "
                "count would be wrong. Edit the script's ResourceConfig instead."
            )
        resources = dataclasses.replace(resources, device=dataclasses.replace(resources.device, variant=config.tpu_type))

    replacements: dict[str, object] = {}
    if config.region is not None:
        replacements["regions"] = (config.region,)
    if config.zone is not None:
        replacements["zone"] = config.zone
    return dataclasses.replace(resources, **replacements) if replacements else resources


def _override_resource_field(value: object, config: LaunchConfig) -> tuple[object, bool]:
    """Override a ``ResourceConfig`` held directly or inside a ``VersionedValue``.

    Returns ``(new_value, changed)``. Only accelerator (TPU/GPU) resources are
    rewritten; CPU resources (inline data-prep steps) are left untouched so their
    cached output paths stay stable. A ``VersionedValue`` wrapper is preserved so
    the resources keep participating in the step's version hash.
    """
    if isinstance(value, VersionedValue):
        resources: object = value.value
        wrapped = True
    else:
        resources = value
        wrapped = False
    if not isinstance(resources, ResourceConfig) or not isinstance(resources.device, TpuConfig | GpuConfig):
        return value, False
    new_resources = override_resources(resources, config)
    if new_resources is resources:
        return value, False
    return (versioned(new_resources) if wrapped else new_resources), True


def _apply_overrides_to_step(step: ExecutorStep, config: LaunchConfig) -> ExecutorStep:
    """Apply ``--tpu_type`` / ``--region`` / ``--zone`` to a step's accelerator resources.

    A step may carry its :class:`ResourceConfig` in two places that must agree:
    ``ExecutorStep.resources`` (used to schedule the Fray job) and the step's
    ``config.resources`` (read at submit time). Either, both, or neither may be
    set, so override both. ``config.resources`` is often wrapped in
    ``versioned(...)``; the wrapper is preserved. Only accelerator resources are
    touched, so inline CPU data-prep steps keep their cached identity.
    """
    replacements: dict[str, object] = {}
    new_step_resources, step_changed = _override_resource_field(step.resources, config)
    if step_changed:
        replacements["resources"] = new_step_resources
    step_config = step.config
    if dataclasses.is_dataclass(step_config) and not isinstance(step_config, type):
        config_fields = {f.name for f in dataclasses.fields(step_config)}
        if "resources" in config_fields:
            new_config_resources, config_changed = _override_resource_field(step_config.resources, config)
            if config_changed:
                replacements["config"] = dataclasses.replace(step_config, resources=new_config_resources)
    return dataclasses.replace(step, **replacements) if replacements else step


def launch_executor(config: LaunchConfig, steps: list[ExecutorStep], description: str | None = None) -> None:
    """Run executor ``steps`` via :func:`launch` — the executor-step entry point.

    Applies any ``--tpu_type`` / ``--region`` / ``--zone`` overrides to each
    step's resources, then runs ``executor_main`` through :func:`launch`. Passing
    ``config.executor`` explicitly bypasses ``executor_main``'s own draccus parse
    (the CLI was already parsed into ``LaunchConfig``).
    """
    if config.tpu_type is not None or config.region is not None or config.zone is not None:
        steps = [_apply_overrides_to_step(step, config) for step in steps]
    launch(config, executor_main, config.executor, steps=steps, description=description)
