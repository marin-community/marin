# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare explicit Mini840 snapshots on CPU before any model service is allocated."""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path

from daytona import AsyncDaytona, CreateSnapshotParams, DaytonaConfig, Image, Resources
from daytona._async.snapshot import SnapshotState

from experiments.evaluation.mini_policy_runtime.environment_inventory import (
    DATASET,
    REVISION,
    publish,
    restored_environments,
)

CONCURRENCY = 4
BUILD_TIMEOUT = 1800


def snapshot_name(source_key):
    digest = hashlib.sha256(source_key.encode()).hexdigest()[:24]
    return "mini840-815c4b2c-" + digest


async def main(smoke: bool, credential_env: str, benchmarks: list[str], data_uri: str):
    logging.basicConfig(level=logging.INFO)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    receipts = []
    failures = []
    with tempfile.TemporaryDirectory() as temporary:
        rows = restored_environments(Path(temporary))
        if benchmarks:
            rows = [row for row in rows if row["benchmark"] in benchmarks]
        if not rows:
            raise ValueError("No matching published Harbor tasks")
        unique = {row["image_source_key"]: row for row in reversed(rows)}
        # Prepare one source for each benchmark first, then the remaining sources.
        leading = {row["benchmark"]: row["image_source_key"] for row in reversed(rows)}
        ordered = list(dict.fromkeys(leading.values() if smoke else [*leading.values(), *unique]))
        quota_exhausted = asyncio.Event()
        async with AsyncDaytona(DaytonaConfig(api_key=os.environ[credential_env])) as client:
            existing = {}
            page = 1
            while True:
                response = await client.snapshot.list(page=page, limit=100)
                existing.update({s.name: s for s in response.items})
                if page >= response.total_pages:
                    break
                page += 1

            async def prepare(key):
                row = unique[key]
                name = snapshot_name(key)
                async with semaphore:
                    if quota_exhausted.is_set():
                        return
                    try:
                        snapshot = existing.get(name)
                        if snapshot is None:
                            image = (
                                Image.base(row["docker_image"])
                                if row["docker_image"]
                                else Image.from_dockerfile(str(Path(row["context"]) / "Dockerfile"))
                            )
                            resources = Resources(
                                cpu=row["cpus"] or 4,
                                memory=math.ceil((row["memory_mb"] or 4096) / 1024),
                                disk=math.ceil((row["storage_mb"] or 10240) / 1024),
                            )
                            async with asyncio.timeout(BUILD_TIMEOUT):
                                snapshot = await client.snapshot.create(
                                    CreateSnapshotParams(name=name, image=image, resources=resources), timeout=0
                                )
                        async with asyncio.timeout(BUILD_TIMEOUT):
                            while snapshot.state != SnapshotState.ACTIVE:
                                if snapshot.state == SnapshotState.ERROR:
                                    raise RuntimeError(f"Snapshot build failed for {name}")
                                await asyncio.sleep(10)
                                snapshot = await client.snapshot.get(name)
                        receipts.append(
                            {
                                "image_source_key": key,
                                "snapshot": name,
                                "snapshot_id": snapshot.id,
                                "image_name": snapshot.image_name,
                                "benchmark": row["benchmark"],
                                "environment_hash": row["environment_hash"],
                                "package_sha256": row["package_sha256"],
                            }
                        )
                        logging.info("Ready %s (%s), %d/%d", name, row["benchmark"], len(receipts), len(unique))
                    except Exception as error:
                        message = str(error).replace(os.environ.get("DAYTONA_API_KEY", "no-key-present"), "[redacted]")
                        if "snapshot quota" in message.lower():
                            quota_exhausted.set()
                        failures.append(
                            {
                                "image_source_key": key,
                                "snapshot": name,
                                "benchmark": row["benchmark"],
                                "error_type": type(error).__name__,
                                "message": message[:1000],
                            }
                        )
                        logging.error("Preparation failed: %s %s", name, type(error).__name__)
                    publish(
                        "environment-preparation-progress.json",
                        {"ready": receipts, "failures": failures, "expected_source_keys": len(unique)},
                    )

            await asyncio.gather(*(prepare(key) for key in ordered))
        report = {
            "dataset": DATASET,
            "revision": REVISION,
            "runtime_commit": "815c4b2c8197bcf1613b405d3cad144dbe0b9f81",
            "credential_env": credential_env,
            "environment_snapshots": {r["image_source_key"]: r["snapshot_id"] for r in receipts},
            "receipts": receipts,
            "failures": failures,
            "expected_source_keys": len(ordered),
            "full_source_keys": len(unique),
            "scope": "smoke" if smoke else "full",
            "required_source_keys": ordered,
            "data_uri": data_uri,
            "tasks": [{"task_id": row["task_id"], "image_source_key": row["image_source_key"]} for row in rows],
            "agentic_tasks": len(rows),
            "auto_build_during_eval": False,
        }
        publish(("smoke-" if smoke else "approved-") + credential_env.lower() + "-environments.json", report)
        if failures:
            raise RuntimeError(f"{len(failures)} of {len(unique)} environments failed CPU preparation")
        print(json.dumps({"approved": len(receipts), "agentic_tasks": len(rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--credential-env", required=True, help="Credential role owning the snapshots; never a secret value."
    )
    parser.add_argument("--data-uri", required=True, help="Preflight dataset directory associated with this map.")
    parser.add_argument("--benchmark", action="append", default=[], help="Explicit diagnostic benchmark subset.")
    parser.add_argument("--smoke", action="store_true", help="Prepare one unchanged task environment per benchmark.")
    args = parser.parse_args()
    asyncio.run(main(args.smoke, args.credential_env, args.benchmark, args.data_uri))
