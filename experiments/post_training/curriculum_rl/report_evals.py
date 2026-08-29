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
@click.option("--date-prefix", default="20260829", show_default=True, help="Eval batch date, bounds the bucket listing.")
def main(date_prefix: str) -> None:
    for arm in ARMS:
        # Layout: evals/<batch>-curriculum-rl-<arm>-<task>-<uid>/results/<task>/<model>/results_*.json
        pattern = f"{EVALS_ROOT}/{date_prefix}-*-curriculum-rl-{arm}-*/results/*/*/results_*.json"
        for child in StoragePath(pattern).glob():
            payload = json.loads(child.read_text())
            for task, metrics in payload.get("results", {}).items():
                slim = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                print(f"== {arm} {task} {json.dumps(slim)}")


if __name__ == "__main__":
    main()
