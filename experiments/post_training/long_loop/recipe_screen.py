# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare or submit one pinned Snowball recipe-screen arm."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import draccus
import jax
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, create_environment
from iris.cli.connect import open_iris_client
from iris.jax.compile_cache import configure_jax_compilation_cache
from iris.rpc.proto_display import priority_band_value
from levanter.checkpoint import load_checkpoint_metadata
from levanter.tracker.json_logger import JsonLoggerConfig  # noqa: F401 -- raw scalar evidence tracker
from marin.training.training import resolve_training_env
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.grug.moe.optimizer import GrugMoeMuonHConfig  # noqa: F401 -- historical optimizer plugin
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.post_training.long_loop.sft_runtime.train import GrugRunConfig, _run_grug_local

SECRET_NAMES = ("CW_KEY_ID", "CW_KEY_SECRET", "WANDB_API_KEY")


def pinned_json(path: Path, sha256: str) -> dict:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError(f"Input hash changed: {path}")
    return json.loads(content)


def train_arm(config: GrugRunConfig, runtime: dict) -> None:
    """Run the existing Grug worker after checking its frozen runtime environment."""
    configure_coreweave_s3()
    configure_jax_compilation_cache()
    for name, expected in runtime["worker_environment"].items():
        if os.environ.get(name) != expected:
            raise ValueError(f"Worker runtime differs: {name}")
    if os.environ.get("XLA_FLAGS") != runtime["xla_flags"]:
        # Iris adds its node-local autotune-cache directory to the frozen flags.
        actual = [
            f
            for f in os.environ.get("XLA_FLAGS", "").split()
            if not f.startswith("--xla_gpu_per_fusion_autotune_cache_dir=")
        ]
        if actual != runtime["xla_flags"].split():
            raise ValueError("Worker XLA flags differ from the reviewed E12 profile")
    output = StoragePath(config.trainer.trainer.checkpointer.base_path).parent
    if int(os.environ["IRIS_MULTIGPU_PROCESS_INDEX"]) == 0:
        (output / "consumed-config.json").write_text(json.dumps(draccus.encode(config), indent=2))
    _run_grug_local(config)
    rank = int(os.environ["IRIS_MULTIGPU_PROCESS_INDEX"])
    (output / f"allocator-end-rank-{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "platform": jax.default_backend(),
                "devices": [{"id": device.id, "stats": device.memory_stats()} for device in jax.local_devices()],
            },
            indent=2,
        )
    )
    step = config.stop_after_steps
    checkpoint = str(output / "checkpoints" / f"step-{step}")
    metadata = load_checkpoint_metadata(checkpoint)
    if metadata["step"] != step or metadata["is_temporary"]:
        raise ValueError("Recipe arm did not commit its selected final checkpoint")
    if int(os.environ["IRIS_MULTIGPU_PROCESS_INDEX"]) == 0:
        (output / "selected.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "checkpoint": checkpoint,
                    "metadata_digest": digest(metadata),
                    "metadata_sha256": (
                        hashlib.sha256((StoragePath(checkpoint) / "metadata.json").read_bytes()).hexdigest()
                    ),
                },
                indent=2,
            )
        )


def request_for_arm(config: GrugRunConfig, runtime: dict, workspace: Path) -> JobRequest:
    """Bind the worker, fixed runtime and independent retry budgets."""
    if config.max_retries_failure or config.max_retries_preemption or config.max_task_failures:
        raise ValueError("Recipe screen requires both retry budgets and max_task_failures zero")
    env = {k: v for k, v in runtime["worker_environment"].items() if v is not None}
    env["XLA_FLAGS"] = runtime["xla_flags"]
    env["JAX_PLATFORMS"] = "cuda,cpu"
    env = resolve_training_env(base_env=env, resources=config.resources)
    return JobRequest(
        name=config.trainer.trainer.id,
        entrypoint=Entrypoint.from_callable(train_arm, args=[config, runtime]),
        resources=config.resources,
        replicas=config.resources.replicas,
        environment=create_environment(workspace=str(workspace), env_vars=env, extras=["gpu"]),
        processes_per_task=config.processes_per_task,
        priority=priority_band_value("production"),
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
        timeout=Duration.from_seconds(config.task_timeout_seconds),
        termination_grace=Duration.from_seconds(60),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    raw = pinned_json(args.config, args.config_sha256)
    runtime = pinned_json(args.runtime, args.runtime_sha256)
    config = draccus.decode(GrugRunConfig, raw)
    workspace = Path(__file__).resolve().parents[3]
    request = request_for_arm(config, runtime, workspace)
    plan = {
        "name": request.name,
        "config_sha256": args.config_sha256,
        "runtime_sha256": args.runtime_sha256,
        "resources": draccus.encode(config.resources),
        "replicas": request.replicas,
        "processes_per_task": request.processes_per_task,
        "failure_retries": request.max_retries_failure,
        "preemption_retries": request.max_retries_preemption,
        "updates": config.stop_after_steps,
        "parameter_policy": str(config.trainer.trainer.mp),
        "optimizer": draccus.encode(config.optimizer),
        "worker_runtime": runtime,
        "gpu_submitted": False,
    }
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(plan, indent=2) + "\n")
    if not args.run:
        return
    secrets = {name: os.environ[name] for name in SECRET_NAMES}
    request.environment.secrets.update(secrets)
    with open_iris_client(config_file=workspace / "lib/iris/config/marin.yaml", workspace=workspace) as client:
        handle = FrayIrisClient.from_iris_client(client).submit(request, adopt_existing=False)
        print(json.dumps({"job_id": str(handle.job_id)}), flush=True)


if __name__ == "__main__":
    main()
