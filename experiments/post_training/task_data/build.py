# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the X and R task splits, collect teacher traces, and sample M1 rows.

    uv run python -m experiments.post_training.task_data.build x --out DIR --cruxeval-prompts FILE
    uv run python -m experiments.post_training.task_data.build r --out DIR --reserved-pool FILE
    uv run python -m experiments.post_training.task_data.build traces --task R --tasks DIR/r_train.jsonl ...
    uv run python -m experiments.post_training.task_data.build m1 --task R --tasks DIR/r_train.jsonl ...

Every command writes a JSON manifest beside its outputs with the inputs, seed,
sampling parameters and drop counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_MODEL
from experiments.post_training.task_data.code_x import exclude_public_cruxeval, synthesize_x_tasks
from experiments.post_training.task_data.code_x import write_split as write_x_split
from experiments.post_training.task_data.expert_outputs import GRADERS, sample_completions, write_m1_rows
from experiments.post_training.task_data.expert_traces import (
    R_COMPLETION_TOKEN_CAP,
    X_COMPLETION_TOKEN_CAP,
    collect_traces,
    family_tokenizer,
    glm_chat_client,
    token_counter,
    write_trace_batch,
)
from experiments.post_training.task_data.math_r import build_r_tasks
from marin.inference.openai_chat import OpenAIChatClient

TOKEN_CAPS = {"X": X_COMPLETION_TOKEN_CAP, "R": R_COMPLETION_TOKEN_CAP}
DATA_SOURCES = {"X": "task_X", "R": "task_R"}
TEACHER_ROUTES = {"X": "E_X", "R": "E_R"}
P_X_SIZE = 1_000


def _read_tasks(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--out", "output_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--cruxeval-prompts", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--train-count", required=True, type=int)
@click.option("--probe-count", default=P_X_SIZE, show_default=True, type=int)
@click.option("--seed", required=True, type=int)
def x(output_dir: Path, cruxeval_prompts: str, train_count: int, probe_count: int, seed: int) -> None:
    """Synthesize the X splits and exclude public CruxEval functions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = synthesize_x_tasks(train_count=train_count, probe_count=probe_count, seed=seed)
    tasks, excluded = exclude_public_cruxeval(tasks, cruxeval_prompts)
    counts = write_x_split(tasks, output_dir)
    _write_manifest(
        output_dir / "x_manifest.json",
        {"seed": seed, "cruxeval_prompts": cruxeval_prompts, "excluded_cruxeval": excluded, "splits": counts},
    )


@cli.command()
@click.option("--out", "output_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--reserved-pool", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--seed", required=True, type=int)
def r(output_dir: Path, reserved_pool: str, seed: int) -> None:
    """Draw the R splits from the pinned pool, excluding the P_think-reserved benchmarks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    build_r_tasks(output_dir, seed=seed, reserved_pool_path=reserved_pool)


@cli.command()
@click.option("--task", "task_name", required=True, type=click.Choice(sorted(TOKEN_CAPS)))
@click.option("--tasks", "tasks_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--family", required=True)
@click.option("--temperature", required=True, type=float)
@click.option("--max-tokens", required=True, type=int)
@click.option("--max-concurrent", required=True, type=int)
@click.option("--relay-job", default=DEFAULT_GLM_RELAY_JOB, show_default=True)
def traces(
    task_name: str,
    tasks_path: str,
    output_path: str,
    family: str,
    temperature: float,
    max_tokens: int,
    max_concurrent: int,
    relay_job: str,
) -> None:
    """Collect GLM teacher traces for building E_X or E_R; over-cap traces are dropped."""
    batch = collect_traces(
        _read_tasks(tasks_path),
        client=glm_chat_client(relay_job),
        model=GLM_MODEL,
        family=family,
        temperature=temperature,
        max_tokens=max_tokens,
        token_cap=TOKEN_CAPS[task_name],
        token_count=token_counter(family_tokenizer(family)),
        max_concurrent=max_concurrent,
    )
    report = write_trace_batch(batch, output_path, temperature=temperature, max_tokens=max_tokens)
    _write_manifest(Path(output_path).with_suffix(".manifest.json"), {"tasks": tasks_path, "model": GLM_MODEL, **report})


@cli.command()
@click.option("--task", "task_name", required=True, type=click.Choice(sorted(TOKEN_CAPS)))
@click.option("--tasks", "tasks_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--family", required=True)
@click.option("--expert-url", required=True, help="OpenAI-compatible base URL of the served expert.")
@click.option("--expert-model", required=True)
@click.option("--expert-token", envvar="EXPERT_API_TOKEN", required=True)
@click.option("--temperature", required=True, type=float)
@click.option("--max-tokens", required=True, type=int)
@click.option("--max-concurrent", required=True, type=int)
@click.option("--timeout", required=True, type=float)
def m1(
    task_name: str,
    tasks_path: str,
    output_path: str,
    family: str,
    expert_url: str,
    expert_model: str,
    expert_token: str,
    temperature: float,
    max_tokens: int,
    max_concurrent: int,
    timeout: float,
) -> None:
    """Sample expert completions and write the graded-correct ones as tokenized M1 rows."""
    tokenizer = family_tokenizer(family)
    batch = sample_completions(
        _read_tasks(tasks_path),
        family=family,
        temperature=temperature,
        max_tokens=max_tokens,
        token_cap=TOKEN_CAPS[task_name],
        client=OpenAIChatClient(expert_url, expert_token, timeout=timeout),
        model=expert_model,
        grade=GRADERS[task_name],
        token_count=token_counter(tokenizer),
        max_concurrent=max_concurrent,
    )
    report = write_m1_rows(
        batch.rows,
        output_path,
        tokenizer=tokenizer,
        family=family,
        data_source=DATA_SOURCES[task_name],
        teacher_route=TEACHER_ROUTES[task_name],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    _write_manifest(
        Path(output_path).with_suffix(".manifest.json"),
        {"tasks": tasks_path, "expert_model": expert_model, "dropped_over_cap": batch.dropped_over_cap, **report},
    )


if __name__ == "__main__":
    cli()
