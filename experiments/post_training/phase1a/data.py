# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lay out each run's M1 rows and M4 prompts in one shared prompt order (control C3).

Both arms read their file with ``data.shuffle=false``. Step ``s`` of M4 rolls out
the step's prompts task by task (8 per 32 sequences, 4 samples each), and step
``s`` of M1 trains on the same prompts' 4 expert completions in the same order. A
prompt is eligible only when its M1 file holds at least 4 graded-correct expert
completions, and the same eligible pool feeds both arms, so the filter cannot
separate them. Each prompt is used once; a shortfall raises.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.post_training.phase1a.matrix import (
    PHASE1A_CONDITIONS,
    SAMPLES_PER_PROMPT,
    SEEDS,
    STEPS,
    TEACHER_ROUTES,
    Condition,
    Method,
    RunPlan,
    Task,
)

M1_FILE = "m1.parquet"
M4_FILE = "m4.parquet"
PROMPT_ONLY_SESSION = "prompt_only"
_M1_COLUMNS = ("task_id", "data_source", "teacher_route", "input_ids", "assistant_masks")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def eligible_samples(m1_rows: list[dict[str, Any]], task: Task) -> dict[str, list[dict[str, Any]]]:
    """Each prompt's first 4 expert completions, for prompts with at least 4."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in m1_rows:
        if row["data_source"] != task or row["teacher_route"] != TEACHER_ROUTES[task]:
            raise ValueError(
                f"M1 row {row['task_id']} is tagged {row['data_source']}/{row['teacher_route']}, not {task}"
            )
        grouped.setdefault(row["task_id"], []).append(row)
    return {
        task_id: rows[:SAMPLES_PER_PROMPT]
        for task_id, rows in sorted(grouped.items())
        if len(rows) >= SAMPLES_PER_PROMPT
    }


def prompt_order(run: RunPlan, eligible: Mapping[Task, Mapping[str, list]]) -> dict[Task, list[str]]:
    """The seeded order of each task's prompts over the run's 100 steps."""
    order = {}
    for task, per_step in run.prompts_per_step.items():
        needed = per_step * STEPS
        pool = sorted(eligible[task])
        if len(pool) < needed:
            raise ValueError(
                f"{run.name} needs {needed} {task} prompts with {SAMPLES_PER_PROMPT} expert samples, has {len(pool)}"
            )
        random.Random(f"{run.seed}:{task}").shuffle(pool)
        order[task] = pool[:needed]
    return order


def layout(
    run: RunPlan, prompts: Mapping[Task, Mapping[str, dict]], samples: Mapping[Task, Mapping[str, list[dict]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """M4 prompt rows and M1 rows, step by step and task by task in condition order."""
    order = prompt_order(run, samples)
    m4_rows: list[dict[str, Any]] = []
    m1_rows: list[dict[str, Any]] = []
    for step in range(STEPS):
        for task, per_step in run.prompts_per_step.items():
            for task_id in order[task][step * per_step : (step + 1) * per_step]:
                m4_rows.append(
                    {
                        "task_id": task_id,
                        "prompt": prompts[task][task_id]["prompt"],
                        "env_class": PROMPT_ONLY_SESSION,
                        "data_source": str(task),
                        "teacher_route": TEACHER_ROUTES[task],
                    }
                )
                m1_rows.extend({column: row[column] for column in _M1_COLUMNS} for row in samples[task][task_id])
    return m4_rows, m1_rows


def build_run_data(
    run: RunPlan, *, task_files: Mapping[Task, Path], m1_files: Mapping[Task, Path], output_dir: Path
) -> dict[str, Any]:
    """Write ``m4.parquet`` and ``m1.parquet`` for one condition and seed; return their counts.

    ``task_files`` hold each task's prompts as WP7 wrote them (``task_id``, ``prompt``);
    ``m1_files`` hold the tokenized expert rows from WP7's ``build m1``.
    """
    prompts = {
        task: {row["task_id"]: {"prompt": [{"role": "user", "content": row["prompt"]}]} for row in _read_jsonl(path)}
        for task, path in task_files.items()
        if task in run.tasks
    }
    samples = {task: eligible_samples(_read_jsonl(m1_files[task]), task) for task in run.tasks}
    m4_rows, m1_rows = layout(run, prompts, samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(m4_rows), output_dir / M4_FILE)
    pq.write_table(pa.Table.from_pylist(m1_rows), output_dir / M1_FILE)
    return {
        "m4_prompts": len(m4_rows),
        "m1_rows": len(m1_rows),
        "eligible_prompts": {str(task): len(samples[task]) for task in run.tasks},
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


_path = click.Path(exists=True, dir_okay=False, path_type=Path)


@click.command()
@click.option("--out", "output_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--x-tasks", required=True, type=_path, help="WP7 x_train.jsonl")
@click.option("--r-tasks", required=True, type=_path, help="WP7 r_train.jsonl")
@click.option("--general-prompts", required=True, type=_path, help="General prompts for the procedure nulls.")
@click.option("--x-m1", required=True, type=_path, help="E_X's tokenized samples on the X prompts.")
@click.option("--r-m1", required=True, type=_path, help="E_R's tokenized samples on the R prompts.")
@click.option("--general-m1", required=True, type=_path, help="B's tokenized samples on the general prompts.")
def main(
    output_dir: Path, x_tasks: Path, r_tasks: Path, general_prompts: Path, x_m1: Path, r_m1: Path, general_m1: Path
) -> None:
    """Write every Phase 1a cell's files and a manifest of their inputs and counts."""
    task_files = {Task.X: x_tasks, Task.R: r_tasks, Task.GENERAL: general_prompts}
    m1_files = {Task.X: x_m1, Task.R: r_m1, Task.GENERAL: general_m1}
    cells = [(condition, seed) for condition in PHASE1A_CONDITIONS for seed in SEEDS]
    cells.append((Condition.PROCEDURE_NULL, SEEDS[0]))
    record: dict[str, Any] = {"inputs": {}, "cells": {}}
    for task in Task:
        record["inputs"][str(task)] = {
            "prompts": {"path": str(task_files[task]), "sha256": _sha256(task_files[task])},
            "m1": {"path": str(m1_files[task]), "sha256": _sha256(m1_files[task])},
        }
    for condition, seed in cells:
        # The method does not change a cell's data; both arms read the same directory.
        run = RunPlan(Method.M1, condition, seed, learning_rate=0.0)
        cell = output_dir / str(condition) / f"seed{seed}"
        counts = build_run_data(run, task_files=task_files, m1_files=m1_files, output_dir=cell)
        record["cells"][f"{condition}/seed{seed}"] = {
            **counts,
            "m4_sha256": _sha256(cell / M4_FILE),
            "m1_sha256": _sha256(cell / M1_FILE),
        }
    (output_dir / "manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
