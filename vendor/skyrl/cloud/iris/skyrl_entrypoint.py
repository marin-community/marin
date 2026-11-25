"""Run a registered SkyRL entrypoint from a composed launch configuration."""

from __future__ import annotations

import argparse
import importlib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, cast

from omegaconf import DictConfig, OmegaConf, open_dict
from iris.cluster.client.job_info import get_job_info
from marinskyrl.resource_locator import is_cloud_uri
from marinskyrl.task_sources import DirectoryDataSource, PreparedTaskCacheSource, data_source
from rigging.filesystem.cluster_config import marin_temp_bucket
from finestore.rl.run_records import (
    AttemptRecord,
    read_attempt,
    write_attempt,
)

from cloud.iris.launch_config import RunMode, derive_launch_topology, load_launch_config
from cloud.iris.rl_config_translation import registered_rl_entrypoint_module


def _prepared_task_cache_source(config: DictConfig) -> PreparedTaskCacheSource | None:
    sources = config.inputs.train_data
    if config.inputs.data_kind != "parquet" or len(sources) != 1:
        return None
    source = data_source(sources[0])
    checkpoint_root = str(config.artifacts.checkpoint_root)
    if not isinstance(source, DirectoryDataSource) or not is_cloud_uri(source.uri) or not is_cloud_uri(checkpoint_root):
        return None
    return PreparedTaskCacheSource(
        path=source.resolved_path(),
        identity=source.identity,
        cache_dir=marin_temp_bucket(14, prefix="marinskyrl/prepared-tasks", source_prefix=checkpoint_root),
    )


def _start_attempt_record(config: DictConfig) -> None:
    """Commit the native receipt and first workload start before training begins."""
    export_root = str(config.artifacts.export_root)
    attempt_id = str(config.run.attempt_id)
    record = read_attempt(export_root, attempt_id)
    try:
        vllm_version = version("vllm")
    except PackageNotFoundError:
        vllm_version = None
    first_start = record is None or record.started_at is None
    facts = {"started_at": datetime.now(timezone.utc).isoformat(), "vllm_version": vllm_version} if first_start else {}
    write_attempt(
        export_root,
        AttemptRecord(
            run_id=str(config.run.id),
            attempt_id=attempt_id,
            iris_job_id=config.skyrl.runtime.iris_job_id,
            config=OmegaConf.to_container(config, resolve=True),
            **facts,
        ),
    )


def run_config(config_path: Path) -> None:
    """Write native startup tracking and execute the configured SkyRL workload."""
    config = load_launch_config(config_path)
    module = registered_rl_entrypoint_module(str(config.runtime.entrypoint))
    job_info = get_job_info()
    with open_dict(config.skyrl):
        config.skyrl.runtime = {
            **config.skyrl.get("runtime", {}),
            "run_id": str(config.run.id),
            **({"model": config.run.model} if config.run.get("model") is not None else {}),
            "attempt_id": str(config.run.attempt_id),
            "coordinator_iris_job_id": config.run.coordinator_iris_job_id,
            "iris_job_id": str(job_info.job_id) if job_info is not None else None,
        }
        if (checkpoint := config.inputs.starting_checkpoint) is not None:
            config.skyrl.runtime.model_source_identity = str(config.inputs.model.identity)
            config.skyrl.runtime.starting_checkpoint_identity = str(checkpoint.identity)
        if config.skyrl.get("trainer", {}).get("mismatch_probe", {}).get("enabled"):
            config.skyrl.runtime.checkpoint_path = str(config.skyrl.trainer.resume_path or config.inputs.model.uri)
    if job_info is not None and config.run.mode != RunMode.CHECKPOINT_EXPORT:
        _start_attempt_record(config)
    run = cast(Callable[[DictConfig], None], getattr(importlib.import_module(module), "run"))
    topology = derive_launch_topology(OmegaConf.to_container(config, resolve=True)).role_topology
    # Benchmark gates compare the recorded launch against the preset before task-local bindings.
    with open_dict(config.skyrl.runtime):
        if config.run.mode == RunMode.TRAIN and (prepared_cache := _prepared_task_cache_source(config)) is not None:
            config.skyrl.runtime.prepared_task_cache = OmegaConf.structured(prepared_cache)
        if topology is not None:
            config.skyrl.runtime.placement = {
                "policy": topology.selector("policy"),
                "engines": [topology.selector(role) for role in topology.roles if role != "policy"],
            }
    run(config.skyrl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_config(args.config)


if __name__ == "__main__":
    main()
