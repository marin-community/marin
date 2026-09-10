# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a validity sample in Daytona sandboxes: the empty workspace, the oracle and the candidate.

    uv run --no-project --isolated --prerelease=allow --with "daytona>=0.182,<1" \\
        python experiments/post_training/tasktrove/validity_daytona.py DIR --jobs 16 \\
        --env TASKTROVE_JUDGE_BASE_URL=... --env TASKTROVE_JUDGE_MODEL=... --env TASKTROVE_JUDGE_API_KEY=...

Runs outside the marin project because the Daytona SDK does not resolve against its lock, so it
imports only the standard library and ``daytona``. ``DAYTONA_API_KEY`` must be set. ``DIR`` is
the output of ``validity sample`` (and ``validity solve``): ``sample.json`` plus one task directory
per task. One Daytona snapshot is built per distinct Dockerfile and reused across the sample's
tasks; when the organisation's snapshot quota refuses a build the run stops, since the
organisation does not allow building a sandbox from a Dockerfile directly (delete stale
snapshots and rerun). Every check runs in a fresh sandbox: ``tests/`` is uploaded to ``/tests``,
the solve directory (``solution/`` for the oracle, ``candidate/`` for the candidate) to
``/solution`` and ``setup_files/`` to ``/setup_files`` when the task ships one,
``/solution/solve.sh`` runs from the image's working directory, then ``/tests/test.sh``
grades and ``/logs/verifier/verdict.json`` comes back as ``logs/<check>/verdict.json``. Checks
that already have a verdict are skipped, and ``results.json`` is rewritten from every verdict on
disk at the end.
"""

import argparse
import json
import logging
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from daytona import (
    CreateSandboxFromSnapshotParams,
    CreateSnapshotParams,
    Daytona,
    DaytonaConfig,
    FileUpload,
    Image,
    Resources,
)
from daytona.common.errors import DaytonaError

logger = logging.getLogger("validity_daytona")

SAMPLE_JSON = "sample.json"
RESULTS_JSON = "results.json"
TASKS_DIR = "tasks"
CHECKS = ("empty", "oracle", "candidate")
SOLVE_DIR_FOR_CHECK = {"oracle": "solution", "candidate": "candidate"}
SOLVE_SH = "solve.sh"
DOCKERFILE = "environment/Dockerfile"
TESTS = "tests"
SETUP_FILES = "setup_files"
TESTS_MOUNT = "/tests"
SOLUTION_MOUNT = "/solution"
SETUP_FILES_MOUNT = "/setup_files"
VERDICT = "/logs/verifier/verdict.json"
DEFAULT_WORKDIR = "/app"
_WORKDIR_LINE = re.compile(r"^WORKDIR\s+(\S+)", re.MULTILINE | re.IGNORECASE)
SNAPSHOT_PREFIX = "tasktrove-validity"
STATUS_NO_SCRIPT = "no_script"
STATUS_NO_VERDICT = "no_verdict"
STATUS_SANDBOX_ERROR = "sandbox_error"
TAIL = 4000


def image_workdir(dockerfile: str) -> str:
    workdirs = _WORKDIR_LINE.findall(dockerfile)
    return workdirs[-1] if workdirs else DEFAULT_WORKDIR


def tree_uploads(root: Path, mount: str) -> list[FileUpload]:
    """Every file under ``root`` as an upload to the same relative path under ``mount``."""
    return [
        FileUpload(source=path.read_bytes(), destination=f"{mount}/{path.relative_to(root)}")
        for path in root.rglob("*")
        if path.is_file()
    ]


def is_quota_error(error: Exception) -> bool:
    text = str(error).lower()
    return "quota" in text or ("limit" in text and "snapshot" in text)


class Snapshots:
    """One Daytona snapshot per Dockerfile id, built on first use."""

    def __init__(self, client: Daytona, resources: Resources, dockerfiles: dict[str, Path]):
        self._client = client
        self._resources = resources
        self._dockerfiles = dockerfiles
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._names: dict[str, str] = {}
        self.created: list[str] = []

    def name_for(self, dockerfile_id: str) -> str:
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
                    if is_quota_error(error):
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


def run_check(
    client: Daytona, snapshots: Snapshots, task_dir: Path, check: str, timeout: int, env: dict[str, str], task: dict
) -> dict:
    """Run one check in a fresh sandbox and write its verdict and command output under ``logs/<check>/``."""
    logs = task_dir / "logs" / check
    logs.mkdir(parents=True, exist_ok=True)
    result = {"name": task["name"], "check": check, "reward": None, "status": "", "detail": "", "solve_exit": None}
    solve_dir = task_dir / SOLVE_DIR_FOR_CHECK[check] if check in SOLVE_DIR_FOR_CHECK else None
    if solve_dir is not None and not (solve_dir / SOLVE_SH).is_file():
        result["status"] = STATUS_NO_SCRIPT
        return result
    dockerfile = task_dir / DOCKERFILE
    workdir = image_workdir(dockerfile.read_text())
    uploads = tree_uploads(task_dir / TESTS, TESTS_MOUNT)
    if (task_dir / SETUP_FILES).is_dir():
        uploads += tree_uploads(task_dir / SETUP_FILES, SETUP_FILES_MOUNT)
    if solve_dir is not None:
        uploads += tree_uploads(solve_dir, SOLUTION_MOUNT)
    sandbox = None
    try:
        params = CreateSandboxFromSnapshotParams(snapshot=snapshots.name_for(task["dockerfile_id"]), ephemeral=True)
        sandbox = client.create(params, timeout=timeout)
        sandbox.fs.upload_files(uploads)
        if solve_dir is not None:
            solve = sandbox.process.exec(f"bash {SOLUTION_MOUNT}/{SOLVE_SH}", cwd=workdir, env=env, timeout=timeout)
            result["solve_exit"] = solve.exit_code
            (logs / "solve.txt").write_text(f"exit {solve.exit_code}\n{solve.result[-TAIL:]}")
        test = sandbox.process.exec(f"bash {TESTS_MOUNT}/test.sh", cwd=workdir, env=env, timeout=timeout)
        (logs / "test.txt").write_text(f"exit {test.exit_code}\n{test.result[-TAIL:]}")
        try:
            verdict = sandbox.fs.download_file(VERDICT)
        except DaytonaError:
            verdict = None
        if not verdict:
            result["status"] = STATUS_NO_VERDICT
            return result
        (logs / "verdict.json").write_bytes(verdict)
        payload = json.loads(verdict)
        result.update(reward=payload["reward"], status=payload["status"], detail=payload["detail"])
        return result
    except DaytonaError as error:
        result["status"] = STATUS_SANDBOX_ERROR
        result["detail"] = f"{type(error).__name__}: {error}"[:TAIL]
        (logs / "error.txt").write_text(result["detail"])
        return result
    finally:
        if sandbox is not None:
            try:
                sandbox.delete()
            except DaytonaError as error:
                logger.warning("could not delete sandbox %s: %s", sandbox.id, error)


def collect_results(root: Path, tasks: list[dict]) -> list[dict]:
    """Every verdict on disk, one row per (task, check)."""
    rows = []
    for task in tasks:
        task_dir = root / TASKS_DIR / task["name"]
        for check in CHECKS:
            verdict = task_dir / "logs" / check / "verdict.json"
            if not verdict.is_file():
                continue
            payload = json.loads(verdict.read_text())
            solve = task_dir / "logs" / check / "solve.txt"
            solve_exit = int(solve.read_text().split("\n", 1)[0].removeprefix("exit ")) if solve.is_file() else None
            rows.append(
                {
                    "name": task["name"],
                    "check": check,
                    "reward": payload["reward"],
                    "status": payload["status"],
                    "detail": payload["detail"],
                    "solve_exit": solve_exit,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--checks", default=",".join(CHECKS))
    parser.add_argument("--timeout", type=int, default=600, help="seconds per sandbox start and per command")
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--memory", type=int, default=2, help="GiB")
    parser.add_argument("--disk", type=int, default=4, help="GiB")
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE forwarded to the solve and test commands")
    parser.add_argument("--keep-snapshots", action="store_true", help="leave the snapshots this run built in place")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    checks = tuple(args.checks.split(","))
    env = dict(item.split("=", 1) for item in args.env)
    tasks = json.loads((args.root / SAMPLE_JSON).read_text())
    dockerfiles = {task["dockerfile_id"]: args.root / TASKS_DIR / task["name"] / DOCKERFILE for task in tasks}
    client = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
    resources = Resources(cpu=args.cpu, memory=args.memory, disk=args.disk)
    snapshots = Snapshots(client, resources, dockerfiles)
    pending = [
        (task, check)
        for task in tasks
        for check in checks
        if not (args.root / TASKS_DIR / task["name"] / "logs" / check / "verdict.json").is_file()
    ]
    logger.info("%d checks to run over %d tasks, %d distinct Dockerfiles", len(pending), len(tasks), len(dockerfiles))

    def run(item: tuple[dict, str]) -> dict:
        task, check = item
        result = run_check(client, snapshots, args.root / TASKS_DIR / task["name"], check, args.timeout, env, task)
        logger.info("%s %s: %s %s", task["name"], check, result["status"], result["reward"])
        return result

    try:
        with ThreadPoolExecutor(args.jobs) as pool:
            list(pool.map(run, pending))
    finally:
        if not args.keep_snapshots:
            snapshots.delete_created()
    results = collect_results(args.root, tasks)
    (args.root / RESULTS_JSON).write_text(json.dumps(results, indent=1))
    scored = [r for r in results if r["status"] == "scored"]
    print(f"{len(results)} verdicts ({len(scored)} scored) written to {args.root / RESULTS_JSON}")


if __name__ == "__main__":
    main()
