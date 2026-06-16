# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-running launcher for experiment scripts.

Marin example scripts (``experiments/grug/*``, ``experiments/tutorials/*``) used
to be started with a two-hop incantation::

    uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \\
      -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.base.launch

``iris job run`` submitted a CPU *launcher* job that ran the script on a worker,
and the script — now inside an Iris task — auto-detected the cluster and
submitted the real training job. The cluster/region/launcher-resources lived in
the hand-typed command line, which is where mistakes accumulated (forgotten or
over-sized launcher resources, wrong region).

This module lets a script hoist the Iris client itself, so it runs directly from
a dev box::

    uv run python experiments/grug/base/launch.py --cluster=marin

:func:`launch_session` resolves the cluster, opens the controller tunnel, and
installs a connected Fray client as the current client; the script's existing
submit path (``executor_main`` or ``current_client().submit``) then targets the
cluster with no launcher job. :func:`override_resources` applies ``--region`` /
``--tpu_type`` to a script's :class:`ResourceConfig`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from fray.current_client import set_current_client
from fray.iris_backend import FrayIrisClient
from fray.types import GpuConfig, ResourceConfig, TpuConfig, get_tpu_topology
from iris.client.connect import connect_to_cluster
from iris.cluster.client.job_info import get_job_info
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main
from marin.execution.types import VersionedValue, versioned

logger = logging.getLogger(__name__)


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


@contextlib.contextmanager
def launch_session(config: LaunchConfig) -> Iterator[None]:
    """Hoist a connected Iris client for the duration of the block, or run locally.

    - Inside an Iris job with ``--cluster`` set: hard error (the user wrapped the
      script in ``iris job run`` *and* asked to re-route — a contradiction).
    - Inside an Iris job without ``--cluster``: the legacy two-hop path. Warn and
      fall back to the auto-detected in-cluster client (back-compat preserved).
    - ``--local`` or no ``--cluster``: yield with no hoist (LocalClient fallback).
    - Otherwise: connect to ``config.cluster`` and install a Fray client as the
      current client; the script's submit path targets the cluster.
    """
    if get_job_info() is not None:
        if config.cluster is not None:
            raise RuntimeError(
                f"--cluster={config.cluster!r} was passed, but this script is already running "
                "inside an Iris job. Self-running examples connect from your dev box: run them "
                "directly (`uv run python <script> --cluster=...`), not via `uv run iris job run`."
            )
        logger.warning(
            "Running inside an Iris job (legacy `uv run iris job run` path). Self-running "
            "examples no longer need that wrapper — run them directly from a dev box. "
            "Using the in-cluster client."
        )
        yield
        return

    if config.local or config.cluster is None:
        logger.info("No --cluster (or --local): running against the in-process LocalClient.")
        yield
        return

    with connect_to_cluster(config.cluster, workspace=_repo_root()) as iris_client:
        with set_current_client(FrayIrisClient.from_iris_client(iris_client)):
            yield


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
    """Convenience for executor-based examples: hoist the client and run ``executor_main``.

    Applies any ``--tpu_type`` / ``--region`` / ``--zone`` overrides to each
    step's resources, then runs ``executor_main`` inside :func:`launch_session`.
    Passing ``config.executor`` explicitly bypasses ``executor_main``'s own
    draccus parse (the CLI was already parsed into ``LaunchConfig``).
    """
    if config.tpu_type is not None or config.region is not None or config.zone is not None:
        steps = [_apply_overrides_to_step(step, config) for step in steps]
    with launch_session(config):
        executor_main(config.executor, steps=steps, description=description)
