# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the X and R task splits, collect teacher traces, and sample M1 rows.

    uv run python -m experiments.post_training.task_data.build x --out DIR --cruxeval-prompts FILE
    uv run python -m experiments.post_training.task_data.build r --out DIR --reserved-pool FILE
    uv run python -m experiments.post_training.task_data.build chunk --tasks DIR/r_train.jsonl --rows 1000 --out DIR
    uv run python -m experiments.post_training.task_data.build traces --task R --tasks DIR/r_train.jsonl ...
    uv run python -m experiments.post_training.task_data.build grade --task R --tasks DIR/r_train.jsonl --traces ...
    uv run python -m experiments.post_training.task_data.build m1 --task R --tasks DIR/r_train.jsonl ...

Every command writes a JSON manifest beside its outputs with the inputs, seed,
sampling parameters and drop counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from marin.inference.openai_chat import OpenAIChatClient

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_MODEL
from experiments.post_training.task_data.code_x import exclude_public_cruxeval, synthesize_x_tasks
from experiments.post_training.task_data.code_x import write_split as write_x_split
from experiments.post_training.task_data.expert_outputs import (
    GRADERS,
    CompletionRow,
    bounded_grade,
    sample_completions,
    write_m1_rows,
)
from experiments.post_training.task_data.expert_traces import (
    R_COMPLETION_TOKEN_CAP,
    X_COMPLETION_TOKEN_CAP,
    ExpertResponse,
    collect_traces,
    family_tokenizer,
    glm_chat_client,
    token_counter,
    write_trace_batch,
)
from experiments.post_training.task_data.math_r import build_r_tasks

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
@click.option("--limit", default=None, type=int, help="cap the pool rows read, to measure the draw rate")
@click.option("--workers", default=1, show_default=True, type=int, help="processes screening the answers")
def r(output_dir: Path, reserved_pool: str, seed: int, limit: int | None, workers: int) -> None:
    """Draw the R splits from the pinned pool, excluding the P_think-reserved benchmarks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    build_r_tasks(output_dir, seed=seed, reserved_pool_path=reserved_pool, limit=limit, workers=workers)


@cli.command()
@click.option("--tasks", "tasks_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--rows", "rows_per_chunk", required=True, type=int)
@click.option("--chunks", "chunk_count", default=None, type=int, help="write only the first N chunks")
@click.option("--out", "output_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
def chunk(tasks_path: str, rows_per_chunk: int, chunk_count: int | None, output_dir: Path) -> None:
    """Write a split's leading rows as fixed-size chunk files, leaving the split untouched."""
    lines = [line for line in Path(tasks_path).read_text().splitlines() if line.strip()]
    starts = range(0, len(lines), rows_per_chunk)
    if chunk_count is not None:
        starts = starts[:chunk_count]
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(tasks_path).stem
    for index, start in enumerate(starts):
        chunk_lines = lines[start : start + rows_per_chunk]
        (output_dir / f"{stem}.chunk{index:02d}.jsonl").write_text("\n".join(chunk_lines) + "\n")
        print(f"{stem}.chunk{index:02d}.jsonl: rows {start}..{start + len(chunk_lines) - 1}")


@cli.command()
@click.option("--task", "task_name", required=True, type=click.Choice(sorted(GRADERS)))
@click.option("--tasks", "tasks_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--traces", "traces_paths", required=True, multiple=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "output_path", required=True, type=click.Path(dir_okay=False))
def grade(task_name: str, tasks_path: str, traces_paths: tuple[str, ...], output_path: str) -> None:
    """Keep the graded-correct teacher traces: the expert training set."""
    tasks = {task["task_id"]: task for task in _read_tasks(tasks_path)}
    traces = [trace for path in traces_paths for trace in _read_tasks(path)]
    grader = GRADERS[task_name]
    verdicts = [bounded_grade(grader, tasks[trace["task_id"]], trace["content"]) for trace in traces]
    # A trace the grader cannot settle (CPU budget or verifier deadline) is not graded correct.
    correct = [trace for trace, verdict in zip(traces, verdicts, strict=True) if verdict]
    with open(output_path, "w") as handle:
        for trace in correct:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
    tokens = sorted(trace["completion_tokens"] for trace in traces)
    report = {
        "tasks": tasks_path,
        "traces": list(traces_paths),
        "kept_under_cap": len(traces),
        "graded_correct": len(correct),
        "ungradable": sum(verdict is None for verdict in verdicts),
        "pass_rate": len(correct) / len(traces) if traces else 0.0,
        "with_reasoning": sum(bool(trace["reasoning"]) for trace in traces),
        "completion_tokens_p50": tokens[len(tokens) // 2] if tokens else None,
        "completion_tokens_p90": tokens[int(len(tokens) * 0.9)] if tokens else None,
        "completion_tokens_max": tokens[-1] if tokens else None,
    }
    _write_manifest(Path(output_path).with_suffix(".manifest.json"), report)
    print(json.dumps(report))


@cli.command()
@click.option("--task", "task_name", required=True, type=click.Choice(sorted(TOKEN_CAPS)))
@click.option("--tasks", "tasks_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--family", required=True)
@click.option("--temperature", required=True, type=float)
@click.option("--max-tokens", required=True, type=int)
@click.option("--max-concurrent", required=True, type=int)
@click.option("--relay-job", default=DEFAULT_GLM_RELAY_JOB, show_default=True)
@click.option("--timeout", required=True, type=float, help="seconds per request; must cover a max-tokens completion")
def traces(
    task_name: str,
    tasks_path: str,
    output_path: str,
    family: str,
    temperature: float,
    max_tokens: int,
    max_concurrent: int,
    relay_job: str,
    timeout: float,
) -> None:
    """Collect GLM teacher traces for building E_X or E_R; over-cap traces are dropped."""
    batch = collect_traces(
        _read_tasks(tasks_path),
        client=glm_chat_client(relay_job, timeout=timeout),
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


# The strong teacher's route on expert-training rows; teacher routes only matter to M4 scoring.
GLM_TEACHER_ROUTE = "glm"


@cli.command("trace-rows")
@click.option("--task", "task_name", required=True, type=click.Choice(sorted(TOKEN_CAPS)))
@click.option("--traces", "traces_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--family", required=True)
@click.option("--temperature", required=True, type=float, help="the traces' sampling temperature, recorded")
@click.option("--max-tokens", required=True, type=int, help="the traces' max_tokens, recorded")
def trace_rows(
    task_name: str, traces_path: str, output_path: str, family: str, temperature: float, max_tokens: int
) -> None:
    """Write graded GLM teacher traces as tokenized rows: the expert-training set for E_X or E_R.

    The rows have the M1 format (WP1's fixed-completion loader), rendered with the
    family's tokenizer through the shared template, reasoning included.
    """
    traces = _read_tasks(traces_path)
    rows = [
        CompletionRow(
            task_id=trace["task_id"],
            prompt=trace["prompt"],
            response=ExpertResponse(reasoning=trace["reasoning"], content=trace["content"], truncated=False),
            graded_correct=True,
        )
        for trace in traces
    ]
    tokenizer = family_tokenizer(family)
    report = write_m1_rows(
        rows,
        output_path,
        tokenizer=tokenizer,
        family=family,
        data_source=DATA_SOURCES[task_name],
        teacher_route=GLM_TEACHER_ROUTE,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    _write_manifest(Path(output_path).with_suffix(".manifest.json"), {"traces": traces_path, **report})


if __name__ == "__main__":
    cli()
