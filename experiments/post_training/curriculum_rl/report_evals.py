# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print aggregated Evalchemy scores for every curriculum-RL arm.

Runs where the experiment bucket is readable (an Iris CPU job on the training
cluster). Scores come from each arm's terminal evaluation artifact directory.
"""

import json

import click
from rigging.filesystem.storage_path import StoragePath

EVALS_ROOT = "s3://marin-us-east-02a/marin/evals"
ARMS = ("naive", "thompson", "grade-uniform", "grade-adaptive", "grade-prior")


@click.command(help=__doc__)
@click.option("--version", default="2026.08.29", show_default=True)
@click.option("--evals", default="math500,gsm8k-0shot", show_default=True)
def main(version: str, evals: str) -> None:
    for arm in ARMS:
        root = StoragePath(f"{EVALS_ROOT}/curriculum-rl-{arm}/{evals}/{version}")
        for child in root.iterdir_recursive():
            name = str(child)
            if not name.endswith(".json"):
                continue
            payload = json.loads(child.read_text())
            print(f"== {arm} {'/'.join(name.rsplit('/', 3)[-2:])}")
            print(json.dumps(payload)[:1500])


if __name__ == "__main__":
    main()
