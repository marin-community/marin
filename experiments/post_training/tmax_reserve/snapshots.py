# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Name and delete the Daytona snapshots of local Harbor task folders. Runs in the vendored Harbor environment.

``names TASK_DIR...`` prints ``{task: snapshot name}`` through Shellbox's own naming path. ``delete NAMES_JSON``
deletes each named snapshot after a fresh ``get``, skipping names that no longer exist, and prints what it did.
``usage`` prints the account's non-general snapshot count and their organization ids. ``existing NAMES_JSON`` prints
which of the named snapshots exist now. ``prepare TASK_DIR...`` builds each distinct snapshot of the folders, one at a
time, and prints ``{name: built name}``.
Only names computed from our own task folders are ever passed to ``delete``.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from daytona import AsyncDaytona, Daytona, DaytonaConfig, DaytonaNotFoundError
from harbor.environments.daytona import machine_spec
from harbor_config.models.task.config import TaskConfig
from shellbox.backends.daytona.machine import DaytonaMachineFactory

from experiments.post_training.task_curation.daytona_identity import context_identity


def names(task_dirs: list[Path]) -> dict[str, str]:
    return {
        task.name: context_identity(task / "environment", (task / "task.toml").read_bytes())
        for task in sorted(task_dirs)
    }


def usage() -> dict:
    """The account's non-general snapshot count (what the quota limits) and the organizations they belong to."""
    client = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
    snapshots, page = [], 1
    while True:
        result = client.snapshot.list(page=page, limit=100)
        snapshots += result.items
        if page >= result.total_pages:
            break
        page += 1
    own = [snapshot for snapshot in snapshots if not snapshot.general]
    return {
        "non_general": len(own),
        "general": len(snapshots) - len(own),
        "organization_ids": sorted({snapshot.organization_id for snapshot in own if snapshot.organization_id}),
    }


def prepare(task_dirs: list[Path]) -> dict[str, str]:
    """Build each distinct snapshot before any trial starts.

    Concurrent first trials of one image otherwise each send ``snapshot.create`` for the same name, and Daytona
    counts those pending creates against the quota (the 200-task null run lost 53 trials that way). Builds run one
    at a time through the factory's own ``prepare``, so trials later only find ACTIVE snapshots.
    """
    config = DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"], target=os.environ["DAYTONA_TARGET"])
    factory = DaytonaMachineFactory(lambda: AsyncDaytona(config))
    specs = {}
    for task, name in names(task_dirs).items():
        task_dir = next(path for path in task_dirs if path.name == task)
        environment = TaskConfig.model_validate_toml((task_dir / "task.toml").read_text()).environment
        specs.setdefault(
            name,
            machine_spec(
                environment,
                task_dir / "environment",
                {},
                cpus=environment.cpus,
                memory_mb=environment.memory_mb,
                storage_mb=environment.storage_mb,
            ),
        )
    built = {name: asyncio.run(factory.prepare(spec)) for name, spec in sorted(specs.items())}
    if any(name != result for name, result in built.items()):
        raise RuntimeError(f"prepare built other names than the naming path gives: {built}")
    return built


def existing(snapshot_names: list[str]) -> list[str]:
    client = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
    found = []
    for name in sorted(set(snapshot_names)):
        try:
            client.snapshot.get(name)
        except DaytonaNotFoundError:
            continue
        found.append(name)
    return found


def delete(snapshot_names: list[str]) -> list[dict]:
    client = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
    log = []
    for name in sorted(set(snapshot_names)):
        try:
            snapshot = client.snapshot.get(name)
        except DaytonaNotFoundError:
            log.append({"name": name, "action": "absent"})
            continue
        if snapshot.name != name or snapshot.general:
            raise ValueError(f"refusing to delete {name}: get returned {snapshot.name} (general={snapshot.general})")
        client.snapshot.delete(snapshot)
        log.append({"name": name, "action": "deleted", "state": str(snapshot.state)})
    return log


def main() -> None:
    command, *arguments = sys.argv[1:]
    if command == "names":
        print(json.dumps(names([Path(arg) for arg in arguments])))
    elif command == "usage":
        print(json.dumps(usage()))
    elif command == "existing":
        print(json.dumps(existing(json.loads(Path(arguments[0]).read_text()))))
    elif command == "prepare":
        print(json.dumps(prepare([Path(arg) for arg in arguments])))
    elif command == "delete":
        print(json.dumps(delete(json.loads(Path(arguments[0]).read_text()))))
    else:
        raise ValueError(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
