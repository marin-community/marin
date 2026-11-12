"""Read curation evidence using Harbor's own models in its locked environment."""

import argparse
import json
import sys
from pathlib import Path

from harbor.models.task.task import Task
from harbor.models.trial.result import TrialResult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("results", "tasks"))
    args = parser.parse_args()
    paths = json.load(sys.stdin)
    inspected = {}
    for name in paths:
        path = Path(name)
        if args.kind == "tasks":
            if Task.is_valid_dir(path):
                task = Task(path)
                inspected[name] = {"task_name": task.name, "task_checksum": task.checksum}
            continue
        document = json.loads(path.read_text())
        if isinstance(document, dict) and "trial_name" in document:
            TrialResult.model_validate(document)
            inspected[name] = True
    json.dump(inspected, sys.stdout)


if __name__ == "__main__":
    main()
