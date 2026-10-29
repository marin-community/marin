# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure Snowball's pass rates on the development panel."""

from collections import defaultdict
from dataclasses import replace

import click
import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl.cli import rl_build_options
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training.snowball_rl_scoring import (
    PASS_RATE_EVALUATION as EVALUATION,
)
from experiments.post_training.snowball_rl_scoring import (
    PASS_RATES_NAME,
    panel_scores,
    read_pass_rates,
    task_manifest,
)


def _write_pass_rates(run_output: str, output: str) -> Artifact:
    rows = panel_scores(EVALUATION, run_output).to_pylist()
    tasks = defaultdict(list)
    for row in rows:
        tasks[row["item"]].append(row)
    expected = set(task_manifest(StoragePath(presets.default_selection().path())))
    if set(tasks) != expected:
        raise ValueError("pass-rate task IDs differ from the pinned development panel")
    if any(len(group) != EVALUATION.answers or not any(row["scored"] for row in group) for group in tasks.values()):
        raise ValueError("pass rates require eight attempted answers and a scored answer for every panel task")
    table = pa.Table.from_pylist(
        [
            {
                "task_id": task,
                "model_revision": presets.MODEL_REVISION,
                "passed": sum(row["scored"] and row["score"] == 1 for row in group),
                "samples": sum(row["scored"] for row in group),
            }
            for task, group in sorted(tasks.items())
        ],
        schema=pa.schema(
            [
                ("task_id", pa.string()),
                ("model_revision", pa.string()),
                ("passed", pa.int32()),
                ("samples", pa.int32()),
            ]
        ),
    )
    fs, key = filesystem_for(str(StoragePath(output) / "pass_rates.parquet"))
    fs.makedirs(key.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(key, "wb") as stream:
        pq.write_table(table, stream)
    read_pass_rates(output)
    return Artifact()


@click.command(help=__doc__)
@rl_build_options
def main() -> tuple[ArtifactStep, ...]:
    preset = replace(presets.PRESETS["data-short"], focus=presets.Focus.LEARNING, evaluations=(EVALUATION,), guards=())
    run = snowball.build_run(preset).runs[0]
    summary = apply(PASS_RATES_NAME, _write_pass_rates, version=presets.PRESET_REVISION, run_output=run, output=OUT)
    click.echo(f"Pass rates: {summary.name}@{summary.version}/pass_rates.parquet")
    return (summary,)


if __name__ == "__main__":
    main()
