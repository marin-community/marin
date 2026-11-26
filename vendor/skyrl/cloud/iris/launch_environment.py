#!/usr/bin/env python3
"""Build task commands and environment for resolved Iris launch configs."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from types import SimpleNamespace

from omegaconf import DictConfig, OmegaConf

from cloud.iris.rl_config_translation import RL_CONFIG_PAYLOAD_ENV
from marinskyrl.environment_contract import DebugMode, EnvVarManager, EnvVarScope
from cloud.iris.runtime_environment import (
    CHECKPOINT_EXPORT_ENTRYPOINT,
    MARINSKYRL_TASK_ROOT,
    RuntimeProfile,
    local_disk_setup_script,
)


def _is_checkpoint_export(args: SimpleNamespace) -> bool:
    return getattr(args, "entrypoint", None) == CHECKPOINT_EXPORT_ENTRYPOINT


@dataclass(frozen=True)
class BundledLaunchConfig:
    """Task path and environment payload for one resolved launch config."""

    task_path: str
    payload: str

    def task_environment(self) -> dict[str, str]:
        """Return the environment needed to materialize the config."""
        return {RL_CONFIG_PAYLOAD_ENV: self.payload}


def build_debug_launch_env(args: SimpleNamespace) -> dict[str, str]:
    """Resolve the effective debug preset after the job name and RL config exist."""
    trainer = args.config.skyrl.get("trainer", {})
    mode = str(trainer.get("debug_mode", DebugMode.LIGHT.value))
    try:
        resolved = DebugMode(mode)
    except ValueError as error:
        choices = ", ".join(item.value for item in DebugMode)
        raise ValueError(f"trainer.debug_mode must be one of: {choices}; got {mode!r}") from error
    if resolved is DebugMode.OFF:
        return {}
    phase_diagnostics = trainer.get("collective_phase_diagnostics") is not False
    return EnvVarManager.for_debug_launch(
        mode=resolved,
        job_name=args.job_name,
        collective_phase_diagnostics=phase_diagnostics,
    ).environment_for(EnvVarScope.TASK_RUNTIME)


def config_task_environment(config: DictConfig) -> dict[str, str]:
    """Resolve task settings and explicitly named backend secrets at submission."""
    extra = dict(config.runtime.task_env)
    skyrl_controls = sorted(str(key) for key in extra if str(key).startswith("SKYRL_"))
    if skyrl_controls:
        raise ValueError(
            "RL config extra_env may not define MarinSkyRL behavior; use typed Hydra settings instead: "
            + ", ".join(skyrl_controls)
        )
    out: dict[str, str] = {}
    for k, v in extra.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = int(v)
        out[str(k)] = str(v)
    if config.runtime.entrypoint != CHECKPOINT_EXPORT_ENTRYPOINT:
        for field in ("remote_judge.api_key_env", "daytona.api_key_env", "daytona.eval_api_key_env"):
            key_env = OmegaConf.select(config, f"skyrl.environment.task_sessions.{field}")
            if key_env is not None:
                value = os.environ.get(key_env)
                if not value:
                    raise ValueError(f"Missing credential environment variable: {key_env}")
                out[key_env] = value
    return out


def _job_scope_fr_dump_path(prefix: str, job_name: str) -> str:
    """Rewrite a JOB-SCOPED NCCL flight-recorder dump path so its slug segment is the
    ACTUAL job name, e.g. ``/tmp/fr_dumps/<slug>/nccl_fr_rank`` -> ``/tmp/fr_dumps/
    <job_name>/nccl_fr_rank``.

    WHY (2026-07-11 FR-slug bug): the 80B configs hardcode
    ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE: /tmp/fr_dumps/80b-next-cp1/nccl_fr_rank`` in
    their ``extra_env:``, so a run launched under a DIFFERENT ``--job-name`` (e.g.
    ``80b-next-cp1-r3d2``) still wrote its FR dumps under the stale ``80b-next-cp1``
    slug (harmless there, but wrong — a future FR dump would land under the wrong
    slug). Deriving the slug from the live job name keeps the dump under the right
    per-job dir. The controller's ``ensure_fr_dump_dir`` mkdir -p's whatever dirname
    the cvar carries, so overriding the cvar here is sufficient.

    ONLY rewrites the job-scoped ``.../fr_dumps/<slug>/<file>`` pattern; a bare
    generic path (``/tmp/nccl_fr_rank``, which every non-80B iris config uses) has no
    slug segment and is returned UNCHANGED (byte-identical for those configs)."""
    parent = os.path.dirname(prefix)  # e.g. /tmp/fr_dumps/<slug>
    grandparent = os.path.dirname(parent)  # e.g. /tmp/fr_dumps
    if os.path.basename(grandparent) != "fr_dumps":
        return prefix  # not a job-scoped fr_dumps path; leave it
    return os.path.join(grandparent, job_name, os.path.basename(prefix))


def build_config_task_command(args: SimpleNamespace) -> list[str]:
    """Build the replica-controller command for the resolved launch config."""
    launch_config = args.launch_config_bundle
    if not isinstance(launch_config, BundledLaunchConfig):
        raise RuntimeError("resolved launch config payload is missing")
    single_domain = args.role_topology is not None and args.role_topology.locality.group_sizes == (args.num_nodes,)
    command = [
        "bash",
        f"{MARINSKYRL_TASK_ROOT}/cloud/iris/node_entrypoint.sh",
        launch_config.task_path,
        f"{args.rendezvous_dir.rstrip('/')}/interconnect-gate",
        *(["--pynccl-init"] if args.runtime_profile is RuntimeProfile.MEGATRON else []),
        *(["--gate-single-nvlink-domain"] if single_domain else []),
    ]
    disk_root = OmegaConf.select(args.config, "skyrl.environment.task_sessions.local.disk_root")
    if disk_root is not None and not _is_checkpoint_export(args):
        return ["bash", "-c", local_disk_setup_script(str(disk_root)) + "\nexec " + shlex.join(command)]
    return command
