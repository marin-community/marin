# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit a frozen format-campaign worker; dry-run unless --submit is explicit.

The standard Iris environment delivers inherited credentials as mounted secrets.
No credential values belong in the launch JSON, dry-run report or receipt.
"""

import argparse
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import draccus
from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling, wrap_multiprocess
from fray.types import GpuConfig, ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from iris.rpc.proto_display import priority_band_value
from levanter.grug.grug_moe import MoeImplementation
from marin.training.training import resolve_training_env
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.grug.moe_hero_ep.model import RematMode
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.train import HERO_EP_RUNTIME_ENV, _apply_hero_ep_runtime_defaults

_SFT_COMMAND = ("python", "-m", "experiments.grug.moe_hero_ep.sft_campaign", "request.json", "--worker")
_ROLLOUT_COMMAND = ("python", "-m", "experiments.grug.moe_hero_ep.rollout_sanity", "run", "request.json")


@dataclass(frozen=True)
class LaunchConfig:
    name: str
    source_revision: str
    controller_cluster: str
    resources: ResourceConfig
    processes_per_task: int
    command: tuple[str, ...]
    request_path: str
    request_sha256: str
    output_root: str
    moe_implementation: MoeImplementation
    remat_mode: RematMode
    cache_prefix: str
    priority: str = "interactive"
    runtime_hours: int = 6
    scheduling_hours: int = 24
    preemption_retries: int = 10
    failure_retries: int = 0

    def __post_init__(self):
        if not isinstance(self.resources.device, GpuConfig):
            raise ValueError("Format-campaign workers require GPU resources")
        if self.resources.device.count != self.processes_per_task:
            raise ValueError("Use one GPU rank per process")
        if self.command == _ROLLOUT_COMMAND and (self.resources.replicas != 16 or self.processes_per_task != 4):
            raise ValueError("Rollout sampling requires 16 tasks with four GPU processes each")
        if self.failure_retries != 0:
            raise ValueError("Application errors require diagnosis, not automatic retry")


@contextmanager
def _restore_environment():
    previous = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def launch_environment(config: LaunchConfig) -> dict[str, str]:
    """Resolve standard hardware policy without persisting inherited secrets."""
    with _restore_environment():
        environment = resolve_training_env(
            base_env={
                "JAX_PLATFORMS": "cuda,cpu",
                "JAX_ENABLE_PGLE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
                "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
                "WANDB_MODE": "disabled",
                "GIT_COMMIT": config.source_revision,
                "MARIN_PREFIX": config.cache_prefix,
            },
            resources=config.resources,
        )
        os.environ.update(environment)
        _apply_hero_ep_runtime_defaults(
            inline_watch_enabled=False,
            processes_per_task=config.processes_per_task,
            moe_implementation=config.moe_implementation,
            remat_mode=config.remat_mode,
        )
        environment.update({name: os.environ[name] for name in (*HERO_EP_RUNTIME_ENV, "XLA_FLAGS")})
        # These belong to the standard mounted-secret inheritance, not ordinary
        # environment serialization. Never include them in the frozen report.
        for name in (
            "HF_TOKEN",
            "WANDB_API_KEY",
            "CW_KEY_ID",
            "CW_KEY_SECRET",
            "CW_ACCESS_KEY_ID",
            "CW_SECRET_ACCESS_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        ):
            environment.pop(name, None)
        return environment


def prepare_launch(config: LaunchConfig) -> tuple[dict, bytes]:
    """Freeze the exact request bytes and expose only nonsecret launch facts."""
    raw = StoragePath(config.request_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config.request_sha256:
        raise ValueError("Worker request bytes changed after review")
    request = json.loads(raw)
    if config.command == _SFT_COMMAND:
        source = request["source_revision"]
        destination = request["output_root"]
    elif config.command == _ROLLOUT_COMMAND:
        source = request["native_request"]["panel"]["source_revision"]
        destination = request["destination"]
    else:
        raise ValueError("Use the supported SFT campaign or rollout worker command")
    if source != config.source_revision:
        raise ValueError("Embedded worker source differs from the launched source revision")
    if destination != config.output_root:
        raise ValueError("Embedded worker destination differs from the guarded output root")
    return {
        "launch_digest": digest(draccus.encode(config)),
        "name": config.name,
        "source_revision": config.source_revision,
        "request_sha256": config.request_sha256,
        "resources": draccus.encode(config.resources),
        "processes_per_task": config.processes_per_task,
        "command": config.command,
        "priority": config.priority,
        "runtime_hours": config.runtime_hours,
        "scheduling_hours": config.scheduling_hours,
        "preemption_retries": config.preemption_retries,
        "failure_retries": config.failure_retries,
        "max_task_failures": 0,
        "submission_requested": False,
    }, raw


def submit_launch(config: LaunchConfig, receipt: Path, repository: Path) -> dict:
    """Submit once; leave an exclusive pending receipt after ambiguous outcomes."""
    summary, raw = prepare_launch(config)
    if receipt.exists() or StoragePath(config.output_root).exists():
        raise FileExistsError("Existing receipt/output: inspect the exact job before any submission")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repository, text=True)
    if actual != config.source_revision or dirty.strip():
        raise ValueError("Submission requires clean tracked source at the exact request revision")
    environment = launch_environment(config)
    native = convert_resources(config.resources)
    entrypoint = Entrypoint(command=list(config.command), workdir_files={"request.json": raw})
    with TemporaryDirectory(prefix="hero-format-source-") as directory:
        snapshot = Path(directory) / "source"
        subprocess.run(["git", "worktree", "add", "--detach", str(snapshot), actual], cwd=repository, check=True)
        try:
            with connect_controller(cluster_name=config.controller_cluster) as endpoint:
                with IrisClient.remote(endpoint.url, credentials=endpoint.credentials, workspace=snapshot) as client:
                    if client.list_jobs(prefix=config.name):
                        raise FileExistsError("An existing job matches the requested identity; inspect it")
                    pending = {**summary, "submission_requested": True, "status": "submission_pending"}
                    # Reserve the receipt before submitting. A lost RPC response
                    # must be recovered by job inspection, never a second submit.
                    with receipt.open("x") as stream:
                        stream.write(json.dumps(pending, indent=2) + "\n")
                    job = client.submit(
                        wrap_multiprocess(entrypoint, native, config.processes_per_task),
                        name=config.name,
                        resources=native,
                        replicas=config.resources.replicas,
                        environment=EnvironmentSpec(env_vars=environment, extras=["gpu"]),
                        constraints=convert_constraints(config.resources),
                        coscheduling=resolve_coscheduling(config.resources, config.resources.replicas),
                        ports=["jax"],
                        timeout=Duration.from_hours(config.runtime_hours),
                        scheduling_timeout=Duration.from_hours(config.scheduling_hours),
                        max_retries_preemption=config.preemption_retries,
                        max_retries_failure=config.failure_retries,
                        max_task_failures=0,
                        priority_band=priority_band_value(config.priority),
                        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
                    )
                    result = {**pending, "status": "submitted", "job_id": str(job.job_id)}
                    # Preserve the pre-submit ambiguity record; the separate
                    # successful receipt is written only after a returned job ID.
                    with receipt.with_suffix(receipt.suffix + ".submitted.json").open("x") as stream:
                        stream.write(json.dumps(result, indent=2) + "\n")
                    return result
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(snapshot)], cwd=repository, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    configure_coreweave_s3()
    config = draccus.decode(LaunchConfig, json.loads(args.launch.read_text()))
    if args.submit:
        if args.receipt is None:
            parser.error("--submit requires an exclusive --receipt path")
        result = submit_launch(config, args.receipt, Path.cwd())
    else:
        result, _ = prepare_launch(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
