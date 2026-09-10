# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end validity sample: a strong model attempts a stratified sample of clean tasks, Daytona grades it.

The Docker samples check that each grader scores an empty workspace 0 and the oracle 1. This
harness asks the harder question, whether a capable model can solve the task from its instruction
alone, so that a group whose tasks nobody can solve stands out as a broken environment rather
than a hard one.

    uv run python -m experiments.post_training.tasktrove.validity sample \\
        --clean s3://marin-us-east-02a/marin/tasktrove/clean/<version> --out DIR --per-group 10
    uv run python -m experiments.post_training.tasktrove.validity solve DIR --model sonnet --jobs 8
    uv run --no-project --isolated --prerelease=allow --with "daytona>=0.182,<1" \\
        python experiments/post_training/tasktrove/validity_daytona.py DIR --jobs 16
    uv run python -m experiments.post_training.tasktrove.validity report DIR

``sample`` picks ``--per-group`` tasks per converter (or mode, or source) from the clean parquet and
writes each as a Harbor task directory under ``DIR/tasks/``, oracle solution included. ``solve``
shows the model the instruction and the Dockerfile and asks for one bash script that completes
the task; the script lands in ``candidate/solve.sh``. The Daytona runner (a separate script,
because the Daytona SDK does not resolve inside this project's lock) grades the empty workspace,
the oracle and the candidate in fresh sandboxes. ``report`` folds the verdicts into per-group
tables. Every step is rerunnable and skips tasks it has already done.
"""

import json
import logging
import os
import random
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, read_task_binary

logger = logging.getLogger(__name__)

SAMPLE_JSON = "sample.json"
RESULTS_JSON = "results.json"
TASKS_DIR = "tasks"
CANDIDATE_DIR = "candidate"
SOLVE_SH = "solve.sh"
RESPONSE_JSON = "response.json"
INDEX_COLUMNS = ("path", "source", "converter", "mode", "dockerfile_id", "has_solution", "tags")
GROUP_KEYS = ("converter", "mode", "source")
CHECKS = ("empty", "oracle", "candidate")
_READERS = 32
_WORKDIR_LINE = re.compile(r"^WORKDIR\s+(\S+)", re.MULTILINE | re.IGNORECASE)
_BASH_BLOCK = re.compile(r"```(?:bash|sh)\n(.*?)```", re.DOTALL)
DEFAULT_WORKDIR = "/app"
DEFAULT_AGENT_COMMAND = "claude -p --output-format json --tools '' --no-session-persistence"
# Env vars that identify the calling agent session; stripped so the solver starts a fresh one
# (the same list the lint review uses).
STRIPPED_ENV = (
    "ANTHROPIC_API_KEY",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_SSE_PORT",
)
PROMPT = """You are solving one sandboxed task without being able to see or touch the sandbox.
Below are the task's instruction and the Dockerfile of the container it runs in. Write ONE bash
script that, run once as root from the container's working directory `{workdir}` with network
access, completes the task: it must leave behind exactly the files the instruction says the grader
reads, with the content the instruction asks for. Work the problem out carefully before writing
the script; the script is your only action and it runs once. If the task needs a repository or
packages, the script must fetch them. Reply with the script in a single ```bash fenced block and
nothing after it.

# Instruction

{instruction}

# Dockerfile

```
{dockerfile}
```
"""


@dataclass(frozen=True)
class SampledTask:
    name: str
    source: str
    path: str
    converter: str
    mode: str
    dockerfile_id: str
    has_solution: bool
    tags: list[str]
    group: str


def stratified_sample(rows: list[dict], group_key: str, per_group: int, seed: int) -> list[dict]:
    """Up to ``per_group`` rows from every value of ``group_key``, drawn with one seeded generator."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r[group_key], r["source"], r["path"])):
        by_group[row[group_key]].append(row)
    rng = random.Random(seed)
    picked = []
    for group in sorted(by_group):
        picked.extend(rng.sample(by_group[group], min(per_group, len(by_group[group]))))
    return picked


def read_index(clean: StoragePath) -> list[dict]:
    """Every clean task's selection columns plus the shard and row it lives in."""

    def read(shard: StoragePath) -> list[dict]:
        with shard.open("rb") as handle:
            table = pq.read_table(handle, columns=list(INDEX_COLUMNS))
        return [{**row, "file": str(shard), "row": i} for i, row in enumerate(table.to_pylist())]

    shards = sorted((clean / TASKS_DIR / "*.parquet").glob(), key=str)
    with ThreadPoolExecutor(_READERS) as pool:
        return [row for rows in pool.map(read, shards) for row in rows]


def export_sample(picks: list[dict], group_key: str, out: Path) -> list[SampledTask]:
    """Write each picked row as a Harbor task directory (oracle solution included) and describe it."""
    by_file: dict[str, list[dict]] = defaultdict(list)
    for pick in picks:
        by_file[pick["file"]].append(pick)

    def export(file: str) -> list[SampledTask]:
        with StoragePath(file).open("rb") as handle:
            table = pq.read_table(handle, columns=["task_binary", "solution_binary"])
        tasks = []
        for pick in by_file[file]:
            name = f"{pick['source']}__{Path(pick['path']).name.removesuffix('.tar.gz')}"
            dest = out / TASKS_DIR / name
            read_task_binary(table.column("task_binary")[pick["row"]].as_py()).write_to(dest)
            solution = table.column("solution_binary")[pick["row"]].as_py()
            if solution is not None:
                read_task_binary(solution).write_to(dest)
            tasks.append(
                SampledTask(
                    name,
                    pick["source"],
                    pick["path"],
                    pick["converter"],
                    pick["mode"],
                    pick["dockerfile_id"],
                    pick["has_solution"],
                    list(pick["tags"]),
                    pick[group_key],
                )
            )
        return tasks

    with ThreadPoolExecutor(_READERS) as pool:
        return sorted((t for ts in pool.map(export, by_file) for t in ts), key=lambda t: (t.group, t.name))


def load_sample(root: Path) -> list[SampledTask]:
    return [SampledTask(**entry) for entry in json.loads((root / SAMPLE_JSON).read_text())]


def image_workdir(dockerfile: str) -> str:
    workdirs = _WORKDIR_LINE.findall(dockerfile)
    return workdirs[-1] if workdirs else DEFAULT_WORKDIR


def script_from_reply(text: str) -> str | None:
    """The last ```bash block of the model's reply, or None when it wrote none."""
    blocks = _BASH_BLOCK.findall(text)
    return blocks[-1] if blocks else None


def solve_task(task_dir: Path, agent_command: list[str], model: str, timeout: float) -> dict:
    """Ask the model for a solve script; write ``candidate/solve.sh`` and the raw reply beside it."""
    dockerfile = (task_dir / DOCKERFILE).read_text()
    prompt = PROMPT.format(
        workdir=image_workdir(dockerfile), instruction=(task_dir / INSTRUCTION).read_text(), dockerfile=dockerfile
    )
    env = {k: v for k, v in os.environ.items() if k not in STRIPPED_ENV}
    proc = subprocess.run(
        [*agent_command, "--model", model], input=prompt, capture_output=True, text=True, timeout=timeout, env=env
    )
    candidate = task_dir / CANDIDATE_DIR
    candidate.mkdir(exist_ok=True)
    if proc.returncode != 0:
        (candidate / "error.txt").write_text(proc.stderr[-4000:] + proc.stdout[-4000:])
        return {"name": task_dir.name, "ok": False, "error": f"exit {proc.returncode}"}
    reply = json.loads(proc.stdout)
    response = {
        "model": model,
        "cost_usd": reply.get("total_cost_usd"),
        "usage": reply.get("usage"),
        "duration_ms": reply.get("duration_ms"),
        "text": reply.get("result", ""),
    }
    (candidate / RESPONSE_JSON).write_text(json.dumps(response, indent=1))
    script = script_from_reply(response["text"])
    if script is None:
        return {"name": task_dir.name, "ok": False, "error": "no bash block in the reply"}
    (candidate / SOLVE_SH).write_text(script)
    return {"name": task_dir.name, "ok": True, "cost_usd": response["cost_usd"]}


def load_results(root: Path) -> dict[tuple[str, str], dict]:
    results = json.loads((root / RESULTS_JSON).read_text()) if (root / RESULTS_JSON).is_file() else []
    return {(r["name"], r["check"]): r for r in results}


def render_report(tasks: list[SampledTask], results: dict[tuple[str, str], dict]) -> str:
    """Per-group counts of what the empty workspace, the oracle and the candidate scored, then every
    task whose verdicts disagree with expectations."""

    def outcome(name: str, check: str) -> str:
        result = results.get((name, check))
        if result is None:
            return "-"
        if result["status"] != "scored":
            return result["status"]
        return "1" if result["reward"] == 1.0 else ("0" if result["reward"] == 0.0 else f"{result['reward']:.2f}")

    by_group: dict[str, list[SampledTask]] = defaultdict(list)
    for task in tasks:
        by_group[task.group].append(task)
    lines = [
        "# Validity sample",
        "",
        f"{len(tasks)} tasks in {len(by_group)} groups. Columns count tasks: `empty=0` and `oracle=1` are the "
        "grader contract; `solved` is a candidate reward of 1; `partial` is strictly between 0 and 1; "
        "`unscored` is a candidate verdict other than scored (invalid task, grader crash, sandbox failure).",
        "",
        "| group | tasks | empty=0 | oracle=1 / with oracle | solved | partial | unscored | mean reward |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, members in sorted(by_group.items()):
        empty_ok = sum(outcome(t.name, "empty") == "0" for t in members)
        with_oracle = [t for t in members if t.has_solution]
        oracle_ok = sum(outcome(t.name, "oracle") == "1" for t in with_oracle)
        scored = [
            results[(t.name, "candidate")]
            for t in members
            if results.get((t.name, "candidate"), {}).get("status") == "scored"
        ]
        solved = sum(r["reward"] == 1.0 for r in scored)
        partial = sum(0.0 < r["reward"] < 1.0 for r in scored)
        unscored = sum((t.name, "candidate") in results for t in members) - len(scored)
        mean = sum(r["reward"] for r in scored) / len(scored) if scored else float("nan")
        lines.append(
            f"| {group} | {len(members)} | {empty_ok} | {oracle_ok} / {len(with_oracle)} | {solved} | {partial} |"
            f" {unscored} | {mean:.2f} |"
        )
    lines += [
        "",
        "## Tasks worth a look",
        "",
        "| task | group | empty | oracle | candidate | candidate detail |",
        "|---|---|---|---|---|---|",
    ]
    for task in tasks:
        empty, oracle, candidate = (outcome(task.name, check) for check in CHECKS)
        expected = empty == "0" and (oracle == "1" or not task.has_solution) and candidate in ("1", "-")
        if expected:
            continue
        detail = results.get((task.name, "candidate"), {}).get("detail", "")
        lines.append(f"| {task.name} | {task.group} | {empty} | {oracle} | {candidate} | {str(detail)[:160]} |")
    return "\n".join(lines) + "\n"


@click.group(help=__doc__)
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    configure_coreweave_s3()


@main.command(help="Draw a stratified sample of clean tasks and export them as task directories.")
@click.option("--clean", required=True, help="clean output, e.g. s3://.../tasktrove/clean/<version>")
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--per-group", type=int, default=10, show_default=True)
@click.option("--group", "group_key", type=click.Choice(GROUP_KEYS), default="converter", show_default=True)
@click.option("--seed", type=int, default=20260910, show_default=True)
def sample(clean: str, out: Path, per_group: int, group_key: str, seed: int) -> None:
    rows = read_index(StoragePath(clean))
    picks = stratified_sample(rows, group_key, per_group, seed)
    logger.info("%d clean tasks, %d picked", len(rows), len(picks))
    tasks = export_sample(picks, group_key, out)
    (out / SAMPLE_JSON).write_text(json.dumps([asdict(t) for t in tasks], indent=1))
    print(f"{len(tasks)} tasks under {out / TASKS_DIR}")


@main.command(help="Ask the model for a solve script for every sampled task that has none yet.")
@click.argument("root", type=click.Path(path_type=Path, exists=True))
@click.option("--model", default="sonnet", show_default=True)
@click.option("--agent-command", default=DEFAULT_AGENT_COMMAND, show_default=True, help="headless agent CLI")
@click.option("--jobs", type=int, default=8, show_default=True)
@click.option("--timeout", type=float, default=900.0, show_default=True, help="seconds per task")
def solve(root: Path, model: str, agent_command: str, jobs: int, timeout: float) -> None:
    pending = [
        root / TASKS_DIR / t.name
        for t in load_sample(root)
        if not (root / TASKS_DIR / t.name / CANDIDATE_DIR / SOLVE_SH).is_file()
    ]
    logger.info("%d tasks to solve", len(pending))
    command = agent_command.split()
    with ThreadPoolExecutor(jobs) as pool:
        outcomes = list(pool.map(lambda d: solve_task(d, command, model, timeout), pending))
    for outcome in outcomes:
        if not outcome["ok"]:
            logger.warning("%s: %s", outcome["name"], outcome["error"])
    cost = sum(o.get("cost_usd") or 0.0 for o in outcomes)
    print(f"{sum(o['ok'] for o in outcomes)} of {len(outcomes)} solved scripts written, ${cost:.2f}")


@main.command(help="Fold the Daytona verdicts into per-group tables; writes report.md beside them.")
@click.argument("root", type=click.Path(path_type=Path, exists=True))
def report(root: Path) -> None:
    text = render_report(load_sample(root), load_results(root))
    (root / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
