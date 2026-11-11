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
from finestore.rl.run_records import (
    AttemptRecord,
    read_attempt,
    run_records_root,
    update_attempt,
)
from finestore.store import DataStore

from cloud.iris.launch_config import RunMode, load_launch_config
from cloud.iris.rl_config_translation import registered_rl_entrypoint_module


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
    with DataStore.open(run_records_root(export_root)) as store:
        update_attempt(
            store,
            AttemptRecord(
                run_id=str(config.run.id),
                attempt_id=attempt_id,
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
            "attempt_id": str(config.run.attempt_id),
            "coordinator_iris_job_id": config.run.coordinator_iris_job_id,
            "iris_job_id": str(job_info.job_id) if job_info is not None else None,
        }
        if config.skyrl.get("trainer", {}).get("mismatch_probe", {}).get("enabled"):
            config.skyrl.runtime.checkpoint_path = str(config.skyrl.trainer.resume_path or config.inputs.model.uri)
    if job_info is not None and config.run.mode != RunMode.CHECKPOINT_EXPORT:
        _start_attempt_record(config)
    run = cast(Callable[[DictConfig], None], getattr(importlib.import_module(module), "run"))
    run(config.skyrl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_config(args.config)


if __name__ == "__main__":
    main()
