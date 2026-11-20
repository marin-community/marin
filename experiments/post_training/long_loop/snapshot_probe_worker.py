# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the shared snapshot pool against real CalibForge task images."""

import argparse
import asyncio
import json
import os
import time
from functools import partial
from pathlib import Path

import ray
from daytona import AsyncDaytona, DaytonaConfig
from harbor.models.task.config import EnvironmentConfig
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, NetworkPolicy
from skyrl_train.rollouts.daytona_snapshot_pool import shared_snapshot_leaser


async def probe(inputs: Path, output: Path) -> None:
    rows = json.loads(inputs.read_text())
    specs = []
    for row in rows:
        cfg = EnvironmentConfig.model_validate(row["environment"])
        specs.append(
            MachineSpec(
                source=RegistryImage(cfg.docker_image),
                workdir="/workspace",
                cpus=cfg.cpus,
                memory_mb=cfg.memory_mb,
                storage_mb=cfg.storage_mb,
                network=NetworkPolicy.ALLOW,
                startup_timeout=900,
            )
        )
    key = os.environ["DAYTONA_RL_API_KEY"]
    url = "https://app.daytona.io/api"
    leaser = shared_snapshot_leaser(capacity=2, api_key=key, api_url=url, target="us")
    factory = DaytonaMachineFactory(
        partial(AsyncDaytona, DaytonaConfig(api_key=key, api_url=url, target="us")),
        snapshot_leaser=leaser,
        create_timeout=900,
        ttl_minutes=30,
    )
    machines = []
    started = time.monotonic()
    report = {"passed": False, "task_ids": [row["id"] for row in rows], "capacity": 2}
    pending = None
    try:
        a1 = await factory.create(specs[0])
        machines.append(a1)
        a2 = await factory.create(specs[0])
        machines.append(a2)
        b = await factory.create(specs[1])
        machines.append(b)
        initial = await leaser.actor.inventory.remote()
        assert len(initial) == 2 and sorted(len(x["leases"]) for x in initial.values()) == [1, 2]
        result = await a1.run(Command(("/bin/sh", "-c", "printf snapshot-probe-ok")))
        assert result.exit_code == 0 and b"snapshot-probe-ok" in result.stdout
        pending = asyncio.create_task(factory.create(specs[2]))
        await asyncio.sleep(2)
        assert not pending.done()
        await a1.close()
        machines.remove(a1)
        assert len(await leaser.actor.inventory.remote()) == 2 and not pending.done()
        await a2.close()
        machines.remove(a2)
        c = await pending
        machines.append(c)
        report["initial_inventory"] = initial
        report["replacement_inventory"] = await leaser.actor.inventory.remote()
        assert len(report["replacement_inventory"]) == 2
        await b.close()
        machines.remove(b)
        await c.close()
        machines.remove(c)
        assert await leaser.actor.inventory.remote() == {}
        report["passed"] = True
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        cleanup = await asyncio.gather(*(m.close() for m in machines), return_exceptions=True)
        report["cleanup_errors"] = [str(x) for x in cleanup if isinstance(x, BaseException)]
        report["final_inventory"] = await leaser.actor.inventory.remote()
        report["elapsed_seconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ray.init(num_cpus=4, include_dashboard=False)
    try:
        asyncio.run(probe(args.inputs, args.output))
    finally:
        ray.shutdown()
