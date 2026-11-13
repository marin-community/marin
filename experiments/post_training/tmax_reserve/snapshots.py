# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Name and delete the Daytona snapshots of local Harbor task folders. Runs in the vendored Harbor environment.

``names TASK_DIR...`` prints ``{task: snapshot name}`` through Shellbox's own naming path. ``delete NAMES_JSON``
deletes each named snapshot after a fresh ``get``, skipping names that no longer exist, and prints what it did.
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
    elif command == "delete":
        print(json.dumps(delete(json.loads(Path(arguments[0]).read_text()))))
    else:
        raise ValueError(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
