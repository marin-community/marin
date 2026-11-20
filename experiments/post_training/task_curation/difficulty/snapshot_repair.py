# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify named snapshots in the RL account without deleting existing snapshots."""

import argparse
import asyncio
import json
from pathlib import Path

from finestore.rl.curation import DAYTONA_SAFE_SNAPSHOT_LIMIT
from harbor.environments.daytona import machine_spec
from harbor.models.task.task import Task
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.snapshot import ensure_snapshot, list_snapshots, snapshot_request_for_spec
from shellbox.image import DockerfileSource

from experiments.post_training.task_curation.difficulty.launch import DAYTONA_API_URL, DAYTONA_TARGET
from experiments.post_training.task_curation.difficulty.planner import BuildSnapshot, action_from_payload
from experiments.post_training.task_curation.difficulty.state import GapReason
from experiments.post_training.task_curation.difficulty_preparation import prepare_source
from experiments.post_training.task_curation.difficulty_samples import load_queue_sources


async def build_snapshots(action: BuildSnapshot, local: Path) -> dict:
    sources = load_queue_sources(action.source.queue_uri, action.source.queue_sha256, (action.source.source_id,))
    prepared = prepare_source(sources[0], local)
    requests = {}
    for selected in prepared.tasks:
        if selected.path is None or selected.reason:
            return {
                "status": "blocked",
                "reason": GapReason.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING,
                "detail": "Original task environment did not pass preparation",
            }
        task = Task(selected.path)
        environment = task.config.environment
        spec = machine_spec(
            environment,
            task.paths.environment_dir,
            {},
            cpus=environment.cpus,
            memory_mb=environment.memory_mb,
            storage_mb=environment.storage_mb,
            gpus=environment.gpus or 0,
        )
        request = snapshot_request_for_spec(spec)
        if isinstance(request.source, DockerfileSource):
            return {
                "status": "blocked",
                "reason": GapReason.ENVIRONMENT_ARTIFACT_MISSING,
                "detail": "RL snapshots require a prepared digest-pinned image; original Dockerfile needs a build",
            }
        requests[request.name] = request
    connection = DaytonaConnection(
        api_key=require_secret("DAYTONA_RL_API_KEY"), api_url=DAYTONA_API_URL, target=DAYTONA_TARGET
    )
    async with connection.client_factory()() as client:
        inventory = await list_snapshots(client)
        names = {snapshot.name for snapshot in inventory}
        missing = set(requests) - names
        if len(names | set(requests)) > DAYTONA_SAFE_SNAPSHOT_LIMIT:
            return {
                "status": "blocked",
                "reason": GapReason.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING,
                "detail": "Snapshot quota needs an owner decision; existing snapshots were preserved",
                "missing": sorted(missing),
            }
        ready = [await ensure_snapshot(client, request) for request in requests.values()]
    return {"status": "ready", "snapshots": ready, "account": "rl"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    action = action_from_payload(json.loads(args.manifest.read_text()))
    if not isinstance(action, BuildSnapshot):
        raise ValueError("Snapshot worker requires a BuildSnapshot action")
    args.result.write_text(json.dumps(asyncio.run(build_snapshots(action, args.local_root)), sort_keys=True))


if __name__ == "__main__":
    main()
