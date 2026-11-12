# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnose the t436 smoke start: same SFT config and 10/05 init path, stop before the data stage.

    python diag_restore.py --plan RESOLVED_PLAN --version V --diag s3://marin-rno2a/tmp/ttl=3d/long-loop-t436/diag/RUN

Run as a CPU job on cw-rno2a. It dispatches one GPU job through the same Fray path as ``run_grug``
(processes_per_task, hero env defaults) with ``replicas`` nodes. Every process:

- writes ``heartbeat/<host>-<pid>.json`` every 10 s (stage, host RSS current and peak), so a SIGKILL
  leaves its last stage on record;
- runs the real ``_run_grug_local`` and stops when it reaches ``build_train_dataset``, which comes
  right after model init, checkpoint restore and the Snowball initializer;
- writes ``result/<host>-<pid>.json`` with ``ok`` or the traceback, plus device memory stats and RSS.
"""

import argparse
import dataclasses
import json
import os
import resource
import socket
import threading
import time
import traceback
from datetime import UTC, datetime

import jax
import psutil
from levanter.trainer import TrainerConfig
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from rigging.timing import Duration

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.post_training.long_loop.execution_policy import DEFAULT_TASK_TIMEOUT_SECONDS
from experiments.post_training.long_loop.sft import fresh_sft_config
from experiments.post_training.long_loop.sft_runtime import train

STAGE = {"name": "process-start"}
STAGED_FUNCTIONS = (
    "compact_grug_mesh",
    "initial_state",
    "restore_template_from",
    "restore_grug_state_from_checkpoint",
    "take_master_as_params",
    "initialize_snowball_sft",
    "dump_grug_state_sharding_run_artifact",
)


class ReachedData(Exception):
    """Raised at build_train_dataset: init and restore finished."""


def write_json(uri: str, value: dict) -> None:
    fs, key = filesystem_for(uri)
    fs.makedirs(fs._parent(key), exist_ok=True)
    fs.pipe_file(key, json.dumps(value, indent=2, default=str).encode())


def host_memory() -> dict:
    process = psutil.Process()
    return {
        "rss_bytes": process.memory_info().rss,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "host_available_bytes": psutil.virtual_memory().available,
        "host_total_bytes": psutil.virtual_memory().total,
    }


def device_memory() -> list[dict]:
    try:
        return [
            {"device": str(device), **{k: v for k, v in (device.memory_stats() or {}).items() if "bytes" in k}}
            for device in jax.local_devices()
        ]
    except Exception as error:
        return [{"error": repr(error)}]


def staged(name: str, fn):
    def wrapper(*args, **kwargs):
        STAGE["name"] = f"{name}:start"
        result = fn(*args, **kwargs)
        STAGE["name"] = f"{name}:done"
        return result

    return wrapper


def diag_entry(payload: tuple) -> None:
    config, diag = payload
    worker = f"{socket.gethostname()}-{os.getpid()}"
    started = time.monotonic()

    def heartbeat() -> None:
        while True:
            write_json(
                prefix_join(diag, f"heartbeat/{worker}.json"),
                {
                    "utc": datetime.now(UTC).isoformat(),
                    "seconds": round(time.monotonic() - started, 1),
                    "stage": STAGE["name"],
                    **host_memory(),
                },
            )
            time.sleep(10)

    threading.Thread(target=heartbeat, daemon=True).start()
    for name in STAGED_FUNCTIONS:
        setattr(train, name, staged(name, getattr(train, name)))
    TrainerConfig.initialize = staged("trainer.initialize", TrainerConfig.initialize)

    def stop(*args, **kwargs):
        STAGE["name"] = "build_train_dataset:reached"
        raise ReachedData

    train.build_train_dataset = stop
    record = {"worker": worker}
    try:
        train._run_grug_local(config)
        record["outcome"] = "returned without reaching build_train_dataset"
    except ReachedData:
        record["outcome"] = "ok: init and restore finished"
    except BaseException as error:
        record["outcome"] = "error"
        record["error"] = repr(error)
        record["traceback"] = traceback.format_exc()
        raise
    finally:
        record.update(
            stage=STAGE["name"],
            seconds=round(time.monotonic() - started, 1),
            host=host_memory(),
            devices=device_memory(),
        )
        write_json(prefix_join(diag, f"result/{worker}.json"), record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="resolved-plan.json of the smoke dry run")
    parser.add_argument("--version", required=True)
    parser.add_argument("--diag", required=True)
    parser.add_argument("--replicas", type=int, default=1)
    args = parser.parse_args()
    if not args.diag.startswith("s3://marin-rno2a/tmp/ttl=3d/long-loop-t436/diag/"):
        raise ValueError("Diagnostics go under s3://marin-rno2a/tmp/ttl=3d/long-loop-t436/diag/")
    with open(args.plan) as plan:
        resolved = json.load(plan)
    name = resolved["name"]
    recipe = {**resolved["sft"], "task_timeout_seconds": DEFAULT_TASK_TIMEOUT_SECONDS}
    config = fresh_sft_config(
        output=prefix_join(args.diag, "sft"),
        initializer=resolved["initializer"]["uri"],
        recipe=recipe,
        run_id=f"{name.replace('/', '-')}-diag-{args.version}",
    )
    resources = dataclasses.replace(config.resources, replicas=args.replicas)
    config = dataclasses.replace(config, resources=resources)
    trainer = config.trainer.trainer
    train._apply_hero_ep_runtime_defaults(
        inline_watch_enabled=trainer.watch.is_enabled and config.trainer.watch_mode == train.WatchMode.INLINE,
        processes_per_task=config.processes_per_task,
        moe_implementation=config.model.moe_implementation,
        remat_mode=config.model.remat_mode,
    )
    write_json(prefix_join(args.diag, "request.json"), {"version": args.version, "replicas": args.replicas})
    dispatch_grug_training_run(
        run_id=trainer.id,
        config=(config, args.diag),
        local_entrypoint=diag_entry,
        resources=resources,
        processes_per_task=config.processes_per_task,
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
        timeout=Duration.from_seconds(3600),
    )


if __name__ == "__main__":
    main()
