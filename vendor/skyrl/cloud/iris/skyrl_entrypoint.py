"""Run a registered SkyRL entrypoint from a composed launch configuration."""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, cast

from omegaconf import DictConfig, OmegaConf, open_dict
from iris.cluster.client.job_info import get_job_info
from marinskyrl.checkpoint_paths import TRACKING_FILENAME
from marinskyrl.resource_locator import join_resource_path
from skyrl_train.io import io

from cloud.iris.launch_config import RunMode, load_launch_config
from cloud.iris.rl_config_translation import registered_rl_entrypoint_module


def run_config(config_path: Path) -> None:
    """Load one launch config and call its registered SkyRL function."""
    config = load_launch_config(config_path)
    module = registered_rl_entrypoint_module(str(config.runtime.entrypoint))
    job_info = get_job_info()
    with open_dict(config.skyrl):
        config.skyrl.runtime = {
            **config.skyrl.get("runtime", {}),
            "run_id": str(config.run.id),
            "attempt_id": str(config.run.attempt_id),
            "coordinator_iris_job_id": config.run.coordinator_iris_job_id,
            "iris_job_id": str(job_info.job_id) if job_info is not None else None,
        }
        if config.skyrl.get("trainer", {}).get("mismatch_probe", {}).get("enabled"):
            config.skyrl.runtime.checkpoint_path = str(config.skyrl.trainer.resume_path or config.inputs.model.uri)
    if job_info is not None and config.run.mode != RunMode.CHECKPOINT_EXPORT:
        directory = join_resource_path(str(config.skyrl.trainer.export_path), "attempts", str(config.run.attempt_id))
        path = join_resource_path(directory, TRACKING_FILENAME)
        previous = json.loads(io.read_bytes(path)) if io.exists(path) else {}
        try:
            vllm_version = version("vllm")
        except PackageNotFoundError:
            vllm_version = None
        generator = OmegaConf.to_container(config.skyrl.generator, resolve=True)
        record = {
            key: config.skyrl.runtime[key] for key in ("run_id", "attempt_id", "iris_job_id", "coordinator_iris_job_id")
        }
        record.update(
            started_at=previous.get("started_at", datetime.now(timezone.utc).isoformat()),
            wandb_url=previous.get("wandb_url"),
            engine={
                **{
                    key: generator.get(key)
                    for key in (
                        "backend",
                        "model_dtype",
                        "vllm_attention_backend",
                        "enable_prefix_caching",
                        "max_num_seqs",
                        "max_num_batched_tokens",
                        "enable_chunked_prefill",
                        "enforce_eager",
                        "engine_init_kwargs",
                    )
                },
                "batch_invariant": config.skyrl.trainer.algorithm.batch_invariant,
                "vllm_env": {key: value for key, value in config.runtime.task_env.items() if key.startswith("VLLM_")},
                "vllm_version": vllm_version,
            },
        )
        io.makedirs(directory, exist_ok=True)
        io.write_bytes_atomic(path, json.dumps(record).encode())
    run = cast(Callable[[DictConfig], None], getattr(importlib.import_module(module), "run"))
    run(config.skyrl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_config(args.config)


if __name__ == "__main__":
    main()
