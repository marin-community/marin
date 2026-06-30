# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers that let an experiment script launch itself onto a cluster.

An experiment script (e.g. ``experiments/grug/base/launch.py``) states its config and
resources inline, builds its lazy :class:`~marin.execution.lazy.ArtifactStep` graph, and
hands a ``body`` to :func:`launch` behind a draccus ``LaunchConfig``. Running *that script*
then chooses where the graph executes::

    uv run python experiments/grug/base/launch.py                    # in-process, LocalClient
    uv run python experiments/grug/base/launch.py --cluster=marin    # submit to the cluster, return
    uv run python experiments/grug/base/launch.py --cluster=marin --follow=true  # ... and stream logs

The resources a run needs live in the script (``ResourceConfig.with_tpu("v5p-8")`` etc.),
so a new user does not have to hand-type ``--cpu`` / ``--memory`` / ``--extra`` /
``--region`` on an ``iris job run`` line. ``--tpu_type`` / ``--region`` / ``--zone`` override
those declared resources at launch.

:func:`launch` runs ``body`` in-process for ``--local``, ``--dry_run``, no ``--cluster``, or
when it is already running inside the coordinator we submitted; otherwise it ships ``body``
to a small CPU *coordinator* job that runs the graph on the cluster and spawns training as
its children. The coordinator outlives this process, so closing the laptop never strands a
run — its ``artifact.json`` / SUCCESS markers are written on the cluster. :func:`run_steps`
is the common ``body``: override resources, then lower and run the handles via
:class:`~marin.execution.step_runner.StepRunner`.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fray.types import ResourceConfig, TpuConfig, get_tpu_topology
from iris.cli.job import add_standard_env_vars, load_env_vars
from iris.client.connect import connect_to_cluster, stream_until_complete
from iris.cluster.client.job_info import get_job_info
from iris.cluster.constraints import Constraint, region_constraint, zone_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from marin.execution.lazy import ArtifactStep, lower
from marin.execution.step_runner import StepRunner

logger = logging.getLogger(__name__)

# The coordinator is a small CPU job that builds the step graph and runs it on the cluster
# (or direct-submits), so the run survives the launching laptop disconnecting. It only
# schedules and dispatches work — CPU is enough, even though building the graph imports
# levanter/marin and the runner holds a thread pool.
COORDINATOR_CPU = 0.5
COORDINATOR_MEMORY = "2GB"
COORDINATOR_DISK = "10GB"

# The runtime-arg key under which a training handle carries the resources it dispatches onto
# (see ``marin.experiment.train.train_lm`` and the grug launchers). Overriding it is how
# ``--tpu_type`` / ``--region`` reach a run without re-fingerprinting it.
TRAIN_RESOURCES_KEY = "train_resources"

# A ``body`` is the script's own work — build handles and run them — invoked as ``body(config)``
# both in-process and on the coordinator. It must be importable on the worker (a top-level
# function) and ``config`` must be cloudpickle-able (a dataclass is).
Body = Callable[["LaunchConfig"], None]


@dataclass
class LaunchConfig:
    """CLI surface for a self-running experiment script, parsed with draccus.

    A script may subclass this to add its own flags (e.g. ``--device`` / ``--dataset``);
    the inherited fields keep working.
    """

    cluster: str | None = None
    """Named Iris cluster to connect to (e.g. ``marin``). When unset, the script runs against
    the in-process LocalClient — no cluster connection."""

    region: str | None = None
    """Restrict submitted training to this region (overrides the script's resources)."""

    zone: str | None = None
    """Restrict submitted training to this zone (overrides the script's resources)."""

    tpu_type: str | None = None
    """Override the script's TPU type (e.g. ``v6e-8``). Must share vm_count with the script
    default; multi-slice / flexible configs must be edited in the script."""

    local: bool = False
    """Force in-process execution even when ``--cluster`` is given."""

    follow: bool = False
    """Stream the coordinator's logs and block until the run finishes. By default the launcher
    returns right after submitting; the coordinator — and the training it spawns — keeps running
    on the cluster regardless, so follow it later with ``iris job logs -f <id>``."""

    dry_run: bool = False
    """Build and lower the graph but do not execute steps (runs in-process)."""


def _repo_root() -> Path:
    """Return the git workspace root to bundle for workers.

    This file lives at ``<root>/experiments/launch.py``, so the root is one parent up; we
    assert ``pyproject.toml`` is there so the bundle is rooted at a real workspace.
    """
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"Expected marin repo root with pyproject.toml at {root}, found none.")
    return root


def launch(config: LaunchConfig, body: Body) -> None:
    """Run ``body(config)`` in-process, or ship it to a coordinator job on the cluster.

    Runs in-process when staying local (``--local``, ``--dry_run``, or no ``--cluster``) or
    when already inside the coordinator we submitted. Otherwise ships ``body`` to a small CPU
    *coordinator* job that runs it on the cluster and spawns training as its children, then
    returns right after submitting (``--follow=true`` instead streams that job's logs and exits
    with its status). Because ``body`` runs on the cluster, the step graph and its SUCCESS
    markers survive the laptop disconnecting.
    """
    if get_job_info() is not None or config.local or config.cluster is None or config.dry_run:
        body(config)
        return
    raise SystemExit(_submit_coordinator_job(config, body))


def run_steps(config: LaunchConfig, *handles: ArtifactStep) -> None:
    """The common ``body``: apply resource overrides, then lower and run ``handles``.

    Pass the training handle(s) a script builds; ``--tpu_type`` / ``--region`` / ``--zone``
    are applied to each before the graph runs.
    """
    overridden = [apply_overrides(config, handle) for handle in handles]
    StepRunner().run([lower(handle) for handle in overridden], dry_run=config.dry_run)


def apply_overrides(config: LaunchConfig, handle: ArtifactStep) -> ArtifactStep:
    """Apply ``--tpu_type`` / ``--region`` / ``--zone`` to a handle's training resources.

    Resources ride on ``runtime_args[TRAIN_RESOURCES_KEY]`` (excluded from the fingerprint), so
    rewriting them never forks the artifact's identity. A handle without that run-arg (e.g. a
    pure data step) is returned unchanged; an override requested against one is a no-op the
    caller is warned about. The other handle fields — and the run-arg mapping's other keys —
    are preserved.
    """
    if config.tpu_type is None and config.region is None and config.zone is None:
        return handle
    resources = handle.runtime_args.get(TRAIN_RESOURCES_KEY)
    if not isinstance(resources, ResourceConfig):
        logger.warning(
            "--tpu_type/--region/--zone given, but %s@%s carries no %r to override; ignoring.",
            handle.name,
            handle.version,
            TRAIN_RESOURCES_KEY,
        )
        return handle
    new_resources = override_resources(config, resources)
    return dataclasses.replace(handle, runtime_args={**handle.runtime_args, TRAIN_RESOURCES_KEY: new_resources})


def override_resources(config: LaunchConfig, resources: ResourceConfig) -> ResourceConfig:
    """Apply ``--tpu_type`` / ``--region`` / ``--zone`` to a :class:`ResourceConfig`.

    Preserves every other scheduling field (replicas, preemptible, image, ...). Rejects unsafe
    TPU swaps — a different ``vm_count`` would silently corrupt the replica count, and a flexible
    (multi-variant) config can't be re-derived from a single ``--tpu_type``. Edit those in the
    script instead.
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


def _coordinator_job_name(body: Body) -> str:
    """A collision-free coordinator job name derived from the body callable's name."""
    label = re.sub(r"[^a-z0-9-]+", "-", body.__name__.lower()).strip("-") or "launch"
    return f"{label}-{uuid.uuid4().hex[:8]}"


def _coordinator_constraints(config: LaunchConfig) -> list[Constraint] | None:
    """Pin the coordinator to ``--region`` / ``--zone`` when one is given.

    The coordinator resolves ``marin_prefix()`` in its own region and the data steps it submits
    inherit its region, so pinning it co-locates the run's storage and accelerators with
    ``--region`` (which also pins the training steps via :func:`override_resources`). When unset,
    the scheduler places the coordinator and each step resolves region-locally on its worker.
    """
    constraints: list[Constraint] = []
    if config.region is not None:
        constraints.append(region_constraint([config.region]))
    if config.zone is not None:
        constraints.append(zone_constraint(config.zone))
    return constraints or None


def _submit_coordinator_job(config: LaunchConfig, body: Body) -> int:
    """Submit a CPU coordinator job that runs ``body(config)`` on the cluster.

    ``body`` and ``config`` are cloudpickled via :meth:`Entrypoint.from_callable`, so the
    coordinator builds and runs the step graph in-cluster and spawns training as its children
    while this process only streams its logs. ``MARIN_PREFIX`` is intentionally not shipped, so
    ``marin_prefix()`` resolves region-locally on the coordinator. Returns the coordinator's exit
    code.
    """
    assert config.cluster is not None
    env_vars = add_standard_env_vars(load_env_vars(None))
    resources = ResourceSpec(cpu=COORDINATOR_CPU, memory=COORDINATOR_MEMORY, disk=COORDINATOR_DISK)
    with connect_to_cluster(config.cluster, workspace=_repo_root()) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_callable(body, config),
            name=_coordinator_job_name(body),
            resources=resources,
            environment=EnvironmentSpec(env_vars=env_vars, extras=["cpu"]),
            constraints=_coordinator_constraints(config),
        )
        # WARNING, not INFO, and with the follow command inline: a bare ``python launch.py``
        # does not configure logging, so only WARNING is visible — and a detached user needs the
        # reconnect command.
        logger.warning(
            "%s submitted to cluster %r as coordinator job %s; it runs on the cluster, not in this "
            "process. Follow it with: iris job logs -f %s",
            body.__name__,
            config.cluster,
            job.job_id,
            job.job_id,
        )
        if not config.follow:
            return 0
        return stream_until_complete(client, job)
