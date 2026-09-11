# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe whether a capable model can solve a stratified sample of clean tasks.

    uv run python -m experiments.post_training.tasktrove.validity sample \\
        --clean s3://marin-us-east-02a/marin/tasktrove/clean/<version> --out DIR --per-group 10
    uv run python -m experiments.post_training.tasktrove.validity solve DIR --model sonnet --jobs 8
    uv run --no-project --isolated --prerelease=allow --with click --with "daytona>=0.182,<1" \\
        python -m experiments.post_training.tasktrove.validity grade DIR --jobs 16
    uv run python -m experiments.post_training.tasktrove.validity report DIR

``sample`` exports task directories, ``solve`` writes model-generated solve scripts, ``grade``
checks empty, oracle, and candidate workspaces in Daytona, and ``report`` summarizes the verdicts.
Every command is rerunnable and skips completed work.
"""

import json
import logging
import os
import random
import re
import shlex
import subprocess
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import batched
from pathlib import Path

import click

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
SOLVE_DIR_FOR_CHECK = {"oracle": "solution", "candidate": CANDIDATE_DIR}
TESTS = "tests"
SETUP_FILES = "setup_files"
TESTS_MOUNT = "/tests"
SOLUTION_MOUNT = "/solution"
SETUP_FILES_MOUNT = "/setup_files"
VERDICT = "/logs/verifier/verdict.json"
SNAPSHOT_PREFIX = "tasktrove-validity"
STATUS_NO_SCRIPT = "no_script"
STATUS_NO_VERDICT = "no_verdict"
STATUS_SANDBOX_ERROR = "sandbox_error"
TAIL = 4000
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


@dataclass(frozen=True)
class SolveResult:
    name: str
    ok: bool
    error: str = ""
    cost_usd: float | None = None


@dataclass(frozen=True)
class ValidityResult:
    name: str
    check: str
    reward: float | None
    status: str
    detail: object = ""
    solve_exit: int | None = None


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


def read_index(clean) -> list[dict]:
    """Every clean task's selection columns plus the shard and row it lives in."""
    import pyarrow.parquet as pq  # noqa: PLC0415 -- sample-only project dependency

    def read(shard) -> list[dict]:
        with shard.open("rb") as handle:
            table = pq.read_table(handle, columns=list(INDEX_COLUMNS))
        return [{**row, "file": str(shard), "row": i} for i, row in enumerate(table.to_pylist())]

    shards = sorted((clean / TASKS_DIR / "*.parquet").glob(), key=str)
    with ThreadPoolExecutor(_READERS) as pool:
        return [row for rows in pool.map(read, shards) for row in rows]


def export_sample(picks: list[dict], group_key: str, out: Path) -> list[SampledTask]:
    """Write each picked row as a Harbor task directory (oracle solution included) and describe it."""
    import pyarrow.parquet as pq  # noqa: PLC0415 -- sample-only project dependency
    from rigging.filesystem.storage_path import StoragePath  # noqa: PLC0415 -- sample-only project dependency

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


def solve_task(task_dir: Path, agent_command: list[str], model: str, timeout: float) -> SolveResult:
    """Ask the model for a solve script; write ``candidate/solve.sh`` and the raw reply beside it."""
    dockerfile = (task_dir / DOCKERFILE).read_text()
    prompt = PROMPT.format(
        workdir=image_workdir(dockerfile), instruction=(task_dir / INSTRUCTION).read_text(), dockerfile=dockerfile
    )
    env = {k: v for k, v in os.environ.items() if k not in STRIPPED_ENV}
    candidate = task_dir / CANDIDATE_DIR
    candidate.mkdir(exist_ok=True)
    try:
        proc = subprocess.run(
            [*agent_command, "--model", model], input=prompt, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired:
        # One hung agent call must not take the rest of the batch down with it.
        (candidate / "error.txt").write_text(f"no reply after {timeout:.0f} s\n")
        return SolveResult(task_dir.name, False, f"timeout after {timeout:.0f} s")
    if proc.returncode != 0:
        (candidate / "error.txt").write_text(proc.stderr[-TAIL:] + proc.stdout[-TAIL:])
        return SolveResult(task_dir.name, False, f"exit {proc.returncode}")
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
        return SolveResult(task_dir.name, False, "no bash block in the reply")
    (candidate / SOLVE_SH).write_text(script)
    return SolveResult(task_dir.name, True, cost_usd=response["cost_usd"])


def load_results(root: Path) -> dict[tuple[str, str], dict]:
    results = json.loads((root / RESULTS_JSON).read_text()) if (root / RESULTS_JSON).is_file() else []
    return {(r["name"], r["check"]): r for r in results}


def tree_uploads(root: Path, mount: str) -> list:
    """Map every file below a task directory to the same relative Daytona path."""
    from daytona import FileUpload  # noqa: PLC0415 -- optional grade dependency

    return [
        FileUpload(source=path.read_bytes(), destination=f"{mount}/{path.relative_to(root)}")
        for path in root.rglob("*")
        if path.is_file()
    ]


def is_snapshot_quota_error(error: Exception) -> bool:
    text = str(error).lower()
    return "quota" in text or ("limit" in text and "snapshot" in text)


class Snapshots:
    """One Daytona snapshot per Dockerfile id, built on first use."""

    def __init__(self, client, resources, dockerfiles: dict[str, Path]):
        self._client = client
        self._resources = resources
        self._dockerfiles = dockerfiles
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._names: dict[str, str] = {}
        self.created: list[str] = []

    def name_for(self, dockerfile_id: str) -> str:
        from daytona import CreateSnapshotParams, Image  # noqa: PLC0415 -- optional grade dependency
        from daytona.common.errors import DaytonaError  # noqa: PLC0415 -- optional grade dependency

        with self._locks[dockerfile_id]:
            if dockerfile_id in self._names:
                return self._names[dockerfile_id]
            name = f"{SNAPSHOT_PREFIX}-{dockerfile_id}"
            try:
                self._client.snapshot.get(name)
                logger.info("snapshot %s exists", name)
            except DaytonaError:
                logger.info("building snapshot %s", name)
                try:
                    self._client.snapshot.create(
                        CreateSnapshotParams(
                            name=name,
                            image=Image.from_dockerfile(self._dockerfiles[dockerfile_id]),
                            resources=self._resources,
                        ),
                        timeout=0,
                    )
                    self.created.append(name)
                except DaytonaError as error:
                    if is_snapshot_quota_error(error):
                        raise RuntimeError(
                            f"snapshot quota is full; delete stale snapshots and rerun ({error})"
                        ) from error
                    raise
            self._names[dockerfile_id] = name
            return name

    def delete_created(self) -> None:
        for name in self.created:
            logger.info("deleting snapshot %s", name)
            self._client.snapshot.delete(name)


def run_daytona_check(
    client, snapshots: Snapshots, task_dir: Path, check: str, timeout: int, env: dict[str, str], task: SampledTask
) -> ValidityResult:
    """Run one validity check in a fresh Daytona sandbox."""
    from daytona import CreateSandboxFromSnapshotParams  # noqa: PLC0415 -- optional grade dependency
    from daytona.common.errors import DaytonaError  # noqa: PLC0415 -- optional grade dependency

    logs = task_dir / "logs" / check
    logs.mkdir(parents=True, exist_ok=True)
    solve_dir = task_dir / SOLVE_DIR_FOR_CHECK[check] if check in SOLVE_DIR_FOR_CHECK else None
    if solve_dir is not None and not (solve_dir / SOLVE_SH).is_file():
        return ValidityResult(task.name, check, None, STATUS_NO_SCRIPT)
    dockerfile = task_dir / DOCKERFILE
    workdir = image_workdir(dockerfile.read_text())
    uploads = tree_uploads(task_dir / TESTS, TESTS_MOUNT)
    if (task_dir / SETUP_FILES).is_dir():
        uploads += tree_uploads(task_dir / SETUP_FILES, SETUP_FILES_MOUNT)
    if solve_dir is not None:
        uploads += tree_uploads(solve_dir, SOLUTION_MOUNT)
    sandbox = None
    try:
        params = CreateSandboxFromSnapshotParams(snapshot=snapshots.name_for(task.dockerfile_id), ephemeral=True)
        sandbox = client.create(params, timeout=timeout)
        sandbox.fs.upload_files(uploads)
        solve_exit = None
        if solve_dir is not None:
            solve_result = sandbox.process.exec(
                f"bash {SOLUTION_MOUNT}/{SOLVE_SH}", cwd=workdir, env=env, timeout=timeout
            )
            solve_exit = solve_result.exit_code
            (logs / "solve.txt").write_text(f"exit {solve_result.exit_code}\n{solve_result.result[-TAIL:]}")
        test_result = sandbox.process.exec(f"bash {TESTS_MOUNT}/test.sh", cwd=workdir, env=env, timeout=timeout)
        (logs / "test.txt").write_text(f"exit {test_result.exit_code}\n{test_result.result[-TAIL:]}")
        try:
            verdict = sandbox.fs.download_file(VERDICT)
        except DaytonaError as error:
            detail = f"could not download {VERDICT}: {error}"[:TAIL]
            (logs / "error.txt").write_text(detail)
            return ValidityResult(task.name, check, None, STATUS_SANDBOX_ERROR, detail, solve_exit)
        if not verdict:
            return ValidityResult(task.name, check, None, STATUS_NO_VERDICT, solve_exit=solve_exit)
        (logs / "verdict.json").write_bytes(verdict)
        payload = json.loads(verdict)
        return ValidityResult(task.name, check, payload["reward"], payload["status"], payload["detail"], solve_exit)
    except DaytonaError as error:
        detail = f"{type(error).__name__}: {error}"[:TAIL]
        (logs / "error.txt").write_text(detail)
        return ValidityResult(task.name, check, None, STATUS_SANDBOX_ERROR, detail)
    finally:
        if sandbox is not None:
            try:
                sandbox.delete()
            except DaytonaError as error:
                logger.warning("could not delete sandbox %s: %s", sandbox.id, error)


def collect_daytona_results(root: Path, tasks: list[SampledTask]) -> list[ValidityResult]:
    rows = []
    for task in tasks:
        task_dir = root / TASKS_DIR / task.name
        for check in CHECKS:
            verdict = task_dir / "logs" / check / "verdict.json"
            if not verdict.is_file():
                continue
            payload = json.loads(verdict.read_text())
            solve = task_dir / "logs" / check / "solve.txt"
            solve_exit = int(solve.read_text().split("\n", 1)[0].removeprefix("exit ")) if solve.is_file() else None
            rows.append(
                ValidityResult(task.name, check, payload["reward"], payload["status"], payload["detail"], solve_exit)
            )
    return rows


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


@main.command(help="Draw a stratified sample of clean tasks and export them as task directories.")
@click.option("--clean", required=True, help="clean output, e.g. s3://.../tasktrove/clean/<version>")
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--per-group", type=int, default=10, show_default=True)
@click.option("--group", "group_key", type=click.Choice(GROUP_KEYS), default="converter", show_default=True)
@click.option("--seed", type=int, default=20260910, show_default=True)
def sample(clean: str, out: Path, per_group: int, group_key: str, seed: int) -> None:
    from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415 -- sample-only dependency
    from rigging.filesystem.storage_path import StoragePath  # noqa: PLC0415 -- sample-only dependency

    configure_coreweave_s3()
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
    command = shlex.split(agent_command)
    with ThreadPoolExecutor(jobs) as pool:
        outcomes = list(pool.map(lambda d: solve_task(d, command, model, timeout), pending))
    for outcome in outcomes:
        if not outcome.ok:
            logger.warning("%s: %s", outcome.name, outcome.error)
    cost = sum(outcome.cost_usd or 0.0 for outcome in outcomes)
    print(f"{sum(outcome.ok for outcome in outcomes)} of {len(outcomes)} solved scripts written, ${cost:.2f}")


@main.command("grade", help="Grade the empty workspace, oracle, and candidate in fresh Daytona sandboxes.")
@click.argument("root", type=click.Path(path_type=Path, exists=True))
@click.option("--jobs", type=int, default=16, show_default=True)
@click.option("--checks", default=",".join(CHECKS), show_default=True)
@click.option("--timeout", type=int, default=600, show_default=True, help="seconds per sandbox start and command")
@click.option("--cpu", type=int, default=1, show_default=True)
@click.option("--memory", type=int, default=2, show_default=True, help="GiB")
@click.option("--disk", type=int, default=4, show_default=True, help="GiB")
@click.option("--env", "environment", multiple=True, help="KEY=VALUE forwarded to solve and test commands")
@click.option("--keep-snapshots", is_flag=True, help="leave snapshots built by this run in place")
@click.option(
    "--images-per-batch",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="maximum Dockerfile snapshots retained concurrently",
)
def grade_sample(
    root: Path,
    jobs: int,
    checks: str,
    timeout: int,
    cpu: int,
    memory: int,
    disk: int,
    environment: tuple[str, ...],
    keep_snapshots: bool,
    images_per_batch: int,
) -> None:
    from daytona import Daytona, DaytonaConfig, Resources  # noqa: PLC0415 -- optional grade dependency

    selected_checks = tuple(checks.split(","))
    unknown = sorted(set(selected_checks) - set(CHECKS))
    if unknown:
        raise click.UsageError(f"unknown checks: {unknown}")
    env = dict(item.split("=", 1) for item in environment)
    tasks = load_sample(root)
    dockerfiles = {task.dockerfile_id: root / TASKS_DIR / task.name / DOCKERFILE for task in tasks}
    client = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
    pending = [
        (task, check)
        for task in tasks
        for check in selected_checks
        if not (root / TASKS_DIR / task.name / "logs" / check / "verdict.json").is_file()
    ]
    logger.info("%d checks over %d tasks and %d images", len(pending), len(tasks), len(dockerfiles))

    def run_one(item: tuple[SampledTask, str]) -> ValidityResult:
        task, check = item
        result = run_daytona_check(client, snapshots, root / TASKS_DIR / task.name, check, timeout, env, task)
        logger.info("%s %s: %s %s", task.name, check, result.status, result.reward)
        return result

    by_image: dict[str, list[tuple[SampledTask, str]]] = defaultdict(list)
    for item in pending:
        by_image[item[0].dockerfile_id].append(item)
    for image_ids in batched(sorted(by_image), images_per_batch):
        image_tasks = [item for image_id in image_ids for item in by_image[image_id]]
        snapshots = Snapshots(
            client,
            Resources(cpu=cpu, memory=memory, disk=disk),
            {image_id: dockerfiles[image_id] for image_id in image_ids},
        )
        logger.info("grading %d checks over %d images", len(image_tasks), len(image_ids))
        try:
            with ThreadPoolExecutor(jobs) as pool:
                list(pool.map(run_one, image_tasks))
        finally:
            if not keep_snapshots:
                snapshots.delete_created()
    results = collect_daytona_results(root, tasks)
    (root / RESULTS_JSON).write_text(json.dumps([asdict(result) for result in results], indent=1))
    scored_results = [result for result in results if result.status == "scored"]
    print(f"{len(results)} verdicts ({len(scored_results)} scored) written to {root / RESULTS_JSON}")


@main.command(help="Fold the Daytona verdicts into per-group tables; writes report.md beside them.")
@click.argument("root", type=click.Path(path_type=Path, exists=True))
def report(root: Path) -> None:
    text = render_report(load_sample(root), load_results(root))
    (root / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
