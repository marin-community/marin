# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Name and delete the Daytona snapshots of local Harbor task folders. Runs in the vendored Harbor environment.

``names TASK_DIR...`` prints ``{task: snapshot name}`` through Shellbox's own naming path. ``delete NAMES_JSON``
deletes each named snapshot after a fresh ``get``, skipping names that no longer exist, and prints what it did.
``usage`` prints the account's non-general snapshot count and their organization ids.
Only names computed from our own task folders are ever passed to ``delete``.
"""

import json
import os
import sys
from pathlib import Path

from daytona import Daytona, DaytonaConfig, DaytonaNotFoundError

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
    elif command == "delete":
        print(json.dumps(delete(json.loads(Path(arguments[0]).read_text()))))
    else:
        raise ValueError(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
