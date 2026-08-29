# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print aggregated Evalchemy metrics for every curriculum-RL arm.

Runs where the experiment bucket is readable (an Iris CPU job on the training
cluster). Reads the canonical per-run ``record.json`` written by the eval
pipeline and filters to this experiment's model names.
"""

import json

import click
from marin.evaluation.records import CW_RECORDS_PREFIX, RunStatus, list_records

from experiments.post_training.curriculum_rl.launch import EXPERIMENT_NAME


@click.command(help=__doc__)
def main() -> None:
    for record in list_records(CW_RECORDS_PREFIX):
        if not record.model.name.startswith(f"{EXPERIMENT_NAME}-"):
            continue
        if record.status is not RunStatus.SUCCEEDED:
            print(f"== {record.model.name} {record.run_id} {record.status}")
            continue
        for task, metrics in record.metrics.items():
            print(f"== {record.model.name} {task} {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
