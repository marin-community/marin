# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Snowball's task experts E_X and E_R: one recipe for both (DESIGN "Experts", C9).

An expert is an M1 run on the task's graded strong-teacher traces (``build trace-rows``):
AdamW at one fixed learning rate after a 5-step warmup, three epochs over the rows in a
seeded shuffle, router level R1, packing off, and one full checkpoint (fp32 masters and
optimizer moments) plus an HF export at the last step. Everything else is Phase 1a's M1 recipe, so the experts train
in the same trainer, precision and clipping as the arms (control C1).

    uv run python -m experiments.post_training.phase1a.experts launch --version YYYY.MM.DD \\
        --learning-rate LR --b-uri S3_URI --b-version V \\
        --rows-uri S3_URI --rows-version V --x-rows N --r-rows N --task task_X --task task_R

``--rows-uri`` is a directory holding ``expert_rows_X.parquet`` and ``expert_rows_R.parquet``
(``python -m experiments.post_training.phase1a.experts rows`` writes them). Without
``--run`` the graph is built and validated only.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import ArtifactDataSource, ArtifactHfModel, SkyRLRun, SkyRLSpec, SkyRLTopology, skyrl_step
from marin.training.training import LevanterCheckpoint

from experiments.post_training.chat_template import FAMILY_TOKENIZER_REVISIONS
from experiments.post_training.phase1a.launch import POLICY_CHAT_TEMPLATE, RETENTION, execution
from experiments.post_training.phase1a.matrix import (
    GPUS_PER_NODE,
    POLICY_NODES,
    Condition,
    Method,
    RunPlan,
    Task,
    role_plan,
    skyrl_recipe,
)

# Snowball's GPU work runs in cw-rno2a, where B's export lives.
EXPERT_CLUSTER = "cw-rno2a"
EXPERT_EPOCHS = 3
EXPERT_SEQUENCES_PER_STEP = 64
EXPERT_SEED = 0
EXPERT_TASKS = (Task.X, Task.R)
_CONDITION = {Task.X: Condition.X, Task.R: Condition.R}
ROWS_FILES = {Task.X: "expert_rows_X.parquet", Task.R: "expert_rows_R.parquet"}
_ROW_COLUMNS = ("task_id", "data_source", "teacher_route", "input_ids", "assistant_masks")


@dataclass(frozen=True)
class ExpertPlan:
    task: Task
    rows: int
    learning_rate: float

    @property
    def name(self) -> str:
        return f"expert-{self.task}-lr{self.learning_rate:g}"

    @property
    def steps(self) -> int:
        """Optimizer steps over three epochs; each epoch drops its last partial batch."""
        steps_per_epoch = self.rows // EXPERT_SEQUENCES_PER_STEP
        if steps_per_epoch == 0:
            raise ValueError(f"{self.task} has {self.rows} rows, fewer than one {EXPERT_SEQUENCES_PER_STEP}-row step")
        return steps_per_epoch * EXPERT_EPOCHS


def expert_recipe(plan: ExpertPlan) -> dict:
    """Phase 1a's M1 recipe with the expert's epochs, shuffle, batch and one final checkpoint.

    The expert saves one full checkpoint (optimizer in DP-local shards plus the FP32
    master records) at its last step; it needs no masters-only schedule.
    """
    recipe = skyrl_recipe(RunPlan(Method.M1, _CONDITION[plan.task], EXPERT_SEED, plan.learning_rate), {})
    trainer = recipe["trainer"]
    trainer.update(
        epochs=EXPERT_EPOCHS,
        max_steps=plan.steps,
        checkpoint_steps=[],
        optimizer_checkpoint_steps=[],
        ckpt_interval=plan.steps,
    )
    recipe["data"]["shuffle"] = True
    return recipe


def expert_topology(plan: ExpertPlan) -> SkyRLTopology:
    base = role_plan(RunPlan(Method.M1, _CONDITION[plan.task], EXPERT_SEED, plan.learning_rate))
    plan_ = dataclasses.replace(
        base, train_batch_size=EXPERT_SEQUENCES_PER_STEP, policy_mini_batch_size=EXPERT_SEQUENCES_PER_STEP
    )
    return SkyRLTopology(num_nodes=POLICY_NODES, gpus_per_node=GPUS_PER_NODE, gpu_variant="H100", role_plan=plan_)


def expert_spec(plan: ExpertPlan, *, model: ArtifactStep[LevanterCheckpoint], rows: ArtifactStep) -> SkyRLSpec:
    repo, revision = FAMILY_TOKENIZER_REVISIONS["grug"]
    name = user_owned_name(f"checkpoints/phase0c/{plan.name}")
    return SkyRLSpec(
        name=name,
        version=resolve_version(name, None),
        config_yaml=yaml.safe_dump(expert_recipe(plan), sort_keys=False),
        model=ArtifactHfModel(
            step=model, tokenizer_uri=repo, tokenizer_revision=revision, chat_template=POLICY_CHAT_TEMPLATE
        ),
        train_data=(ArtifactDataSource(rows, relative_path=ROWS_FILES[plan.task]),),
        validation_data=(),
        topology=expert_topology(plan),
        retention=RETENTION,
        seed=EXPERT_SEED,
    )


def write_rows_parquet(jsonl_path: Path, parquet_path: Path) -> int:
    """The tokenized rows of ``build trace-rows`` as the fixed-completion loader's parquet."""
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    pq.write_table(
        pa.Table.from_pylist([{column: row[column] for column in _ROW_COLUMNS} for row in rows]), parquet_path
    )
    return len(rows)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--x-rows", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--r-rows", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "output_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
def rows(x_rows: Path, r_rows: Path, output_dir: Path) -> None:
    """Write both experts' training rows as parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for task, path in ((Task.X, x_rows), (Task.R, r_rows)):
        count = write_rows_parquet(path, output_dir / ROWS_FILES[task])
        click.echo(f"{ROWS_FILES[task]}: {count} rows")


@cli.command(help=__doc__)
@click.option("--learning-rate", required=True, type=float)
@click.option("--b-uri", required=True, help="Stage 0 Snowball B: the artifact root holding its HF export.")
@click.option("--b-version", required=True)
@click.option("--rows-uri", required=True)
@click.option("--rows-version", required=True)
@click.option("--x-rows", "x_row_count", required=True, type=int, help="row count of expert_rows_X.parquet")
@click.option("--r-rows", "r_row_count", required=True, type=int, help="row count of expert_rows_R.parquet")
@click.option("--task", "tasks", multiple=True, required=True, type=click.Choice([str(task) for task in EXPERT_TASKS]))
@rl_build_options
def launch(
    learning_rate: float,
    b_uri: str,
    b_version: str,
    rows_uri: str,
    rows_version: str,
    x_row_count: int,
    r_row_count: int,
    tasks: tuple[str, ...],
) -> list[ArtifactStep[SkyRLRun]]:
    model = ArtifactStep.adopt(user_owned_name("models/phase1a-snowball-b"), b_version, b_uri, kind=LevanterCheckpoint)
    rows_step = ArtifactStep.adopt(user_owned_name("documents/phase0c-expert-rows"), rows_version, rows_uri)
    counts = {Task.X: x_row_count, Task.R: r_row_count}
    return [
        skyrl_step(
            expert_spec(ExpertPlan(Task(task), counts[Task(task)], learning_rate), model=model, rows=rows_step),
            execution(EXPERT_CLUSTER),
            export_hf=True,
        )
        for task in tasks
    ]


if __name__ == "__main__":
    cli()
