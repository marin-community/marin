# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert a sample of one source and run its graders in Docker, the way Harbor would.

    uv run python -m experiments.post_training.tasktrove.sample_run \\
        --source laion__nemotron-gym-knowledge-mcqa-v2 --count 20 --out /tmp/sample

For every converted task the empty check runs the shim in a fresh workspace and must score 0.
When the task ships an oracle solution, the oracle check runs ``solution/solve.sh`` in the
workspace first and must score 1. The tool is installed from the local checkout rather than the
git ref in the Dockerfile, so a converter can be checked before its branch is pushed. Converter
authors run this while writing a converter and commit the JSON report it prints under
``converters/reports/``.
"""

import collections
import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import click
from tasktrove_verify.cli import DEFAULT_LOGS_DIR
from tasktrove_verify.reward import REWARD_JSON
from zephyr.readers import load_parquet

from experiments.post_training.tasktrove.contract import INSTALL_MARKER, TESTS_MOUNT
from experiments.post_training.tasktrove.convert import ConvertedRecord, convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.sources import load_source_verdicts
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, read_task_binary

logger = logging.getLogger(__name__)

LOCAL_TOOL_REF = "local"
_TOOL_DIR = Path(__file__).parents[3] / "lib" / "tasktrove-verify"
_INSTALL_LINE = re.compile(
    r'^RUN (UV_TOOL_BIN_DIR=\S+ )?uv tool install (--python "[^"]+" )?"tasktrove-verify(\[[^\]]*\])? @ [^"]+"$',
    re.MULTILINE,
)
_WORKDIR_LINE = re.compile(r"^WORKDIR\s+(\S+)", re.MULTILINE | re.IGNORECASE)
SETUP_FILES_DIR = "setup_files"
NO_NETWORK = "none"


@dataclass(frozen=True)
class CheckResult:
    path: str
    check: str
    reward: float | None
    status: str
    detail: str


@dataclass(frozen=True)
class SampleReport:
    source: str
    converter: str
    sampled: int
    statuses: dict[str, int]
    images: int
    checks: list[CheckResult]

    @property
    def ok(self) -> bool:
        return all(
            (c.check == "empty" and c.reward == 0.0 and c.status == "scored")
            or (c.check == "oracle" and c.reward == 1.0 and c.status == "scored")
            for c in self.checks
        )


def local_dockerfile(dockerfile: str) -> str:
    """The task's Dockerfile with the git install swapped for a copy of the local checkout."""
    if INSTALL_MARKER not in dockerfile:
        raise ValueError("Dockerfile has no tool install block")
    replaced, count = _INSTALL_LINE.subn(
        lambda m: (
            "COPY tasktrove-verify /opt/tasktrove-verify\n"
            f'RUN {m.group(1) or ""}uv tool install {m.group(2) or ""}"/opt/tasktrove-verify{m.group(3) or ""}"'
        ),
        dockerfile,
    )
    if count != 1:
        raise ValueError("Dockerfile does not carry exactly one tool install line")
    return replaced


def build_image(dockerfile: str, tag: str) -> None:
    with tempfile.TemporaryDirectory(prefix="tasktrove-build-") as context:
        shutil.copytree(
            _TOOL_DIR, Path(context) / "tasktrove-verify", ignore=shutil.ignore_patterns(".venv", "__pycache__")
        )
        (Path(context) / "Dockerfile").write_text(local_dockerfile(dockerfile))
        subprocess.run(["docker", "build", "-q", "-t", tag, context], check=True, capture_output=True, text=True)


def image_workdir(dockerfile: str) -> str:
    """The last ``WORKDIR`` the Dockerfile sets, where the oracle runs; ``/app`` when it sets none."""
    workdirs = _WORKDIR_LINE.findall(dockerfile)
    return workdirs[-1] if workdirs else "/app"


def run_check(image: str, task_dir: Path, check: str, timeout: float, network: str) -> CheckResult:
    logs = task_dir / "logs" / check
    logs.mkdir(parents=True, exist_ok=True)
    workdir = image_workdir((task_dir / "environment" / "Dockerfile").read_text())
    command = f"bash {TESTS_MOUNT}/test.sh"
    mounts = ["-v", f"{task_dir / 'tests'}:{TESTS_MOUNT}:ro", "-v", f"{logs}:{DEFAULT_LOGS_DIR}"]
    if (task_dir / SETUP_FILES_DIR).is_dir():
        mounts += ["-v", f"{task_dir / SETUP_FILES_DIR}:/{SETUP_FILES_DIR}:ro"]
    if check == "oracle":
        mounts += ["-v", f"{task_dir / 'solution'}:/solution:ro"]
        command = f"cd {workdir} && bash /solution/solve.sh && bash {TESTS_MOUNT}/test.sh"
    argv = ["docker", "run", "--rm", "--network", network, *mounts, image, "bash", "-c", command]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return CheckResult(task_dir.name, check, None, "timeout", "")
    reward_file = logs / REWARD_JSON
    if not reward_file.is_file():
        return CheckResult(task_dir.name, check, None, "no_reward", (proc.stderr or proc.stdout)[-2000:])
    payload = json.loads(reward_file.read_text())
    return CheckResult(task_dir.name, check, payload["reward"], payload["status"], json.dumps(payload["detail"])[:500])


def write_task_dir(record: ConvertedRecord, root: Path) -> Path:
    task_dir = root / Path(record.path).name.removesuffix(".tar.gz")
    read_task_binary(record.task_binary).write_to(task_dir)
    if record.solution_binary is not None:
        read_task_binary(record.solution_binary).write_to(task_dir)
    return task_dir


def sample_source(source: str, parquet: Path, count: int, out: Path, timeout: float, network: str) -> SampleReport:
    info = load_source_verdicts()[source]
    index = converter_index()
    statuses: collections.Counter = collections.Counter()
    converted: list[ConvertedRecord] = []
    for row in load_parquet(str(parquet)):
        if len(converted) >= count:
            break
        record = convert_one(info, row["path"], row["task_binary"], index, LOCAL_TOOL_REF)
        statuses[record.status] += 1
        if record.status == ConvertStatus.CONVERTED:
            converted.append(record)
    images: dict[str, str] = {}
    checks: list[CheckResult] = []
    for record in converted:
        task_dir = write_task_dir(record, out)
        tag = images.get(record.dockerfile_id)
        if tag is None:
            tag = f"tasktrove-sample:{record.dockerfile_id}"
            logger.info("building %s", tag)
            build_image(read_task_binary(record.task_binary).text(DOCKERFILE), tag)
            images[record.dockerfile_id] = tag
        checks.append(run_check(tag, task_dir, "empty", timeout, network))
        if record.solution_binary is not None:
            checks.append(run_check(tag, task_dir, "oracle", timeout, network))
    converters = {r.converter for r in converted}
    return SampleReport(source, ",".join(sorted(converters)), len(converted), dict(statuses), len(images), checks)


@click.command(help=__doc__)
@click.option("--source", required=True)
@click.option(
    "--parquet",
    type=click.Path(path_type=Path),
    default=None,
    help="defaults to ~/data/tasktrove/repo/<source>/tasks.parquet",
)
@click.option("--count", type=int, default=20)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--timeout", type=float, default=600.0, help="seconds per container run")
@click.option(
    "--network",
    default=NO_NETWORK,
    show_default=True,
    help="docker network for the checks; SWE oracles clone their repository and need 'bridge'",
)
def main(source: str, parquet: Path | None, count: int, out: Path, timeout: float, network: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parquet = parquet or Path.home() / "data" / "tasktrove" / "repo" / source / "tasks.parquet"
    report = sample_source(source, parquet, count, out, timeout, network)
    print(json.dumps(asdict(report), indent=1))
    print(f"{report.sampled} tasks, {report.images} images, {len(report.checks)} checks, ok={report.ok}")
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
