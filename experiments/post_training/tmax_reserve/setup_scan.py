# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find the pilot tasks whose post_install.sh has a failing step, under local Docker.

Many post_install.sh scripts lack ``set -e``, so a failed step (a fetch, an apt or pip install) still lets the
image build, and the task ships partly broken. For each task this runs its ``task_setup/`` on its shared image, as
the trial does, but with an ``ERR`` trap that logs every failing command with its line and exit status. A step
inside ``if``, ``&&`` or ``||`` does not count (the script handles it); a failure inside a pipeline counts only
when it is the last command, as bash has no ``pipefail`` here.

The two shared images are built once from their template contexts with a dedicated buildx builder; images,
containers and the builder are removed at the end. CPU only. Writes one JSON row per task to ``--output``.
"""

import argparse
import json
import logging
import re
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.post_training.tmax_reserve.shared_image import TMUX_STEP, BuildSteps, Template

logger = logging.getLogger(__name__)

BUILDER = "tmax-setup-scan"
BUILDKIT_IMAGE = "moby/buildkit:buildx-stable-1"
GUEST = "/tmp/.tmax-setup-scan"
FAILED_MARK = "TMAX_FAILED_STEP"
FAILED_LINE = re.compile(rf"{FAILED_MARK} line (\d+) exit (\d+): (.*)")
SETUP_TIMEOUT = 1800


def docker(
    *argv: str, check: bool = True, timeout: float | None = None, stdin: str | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *argv], capture_output=True, text=True, check=check, timeout=timeout, input=stdin)


def traced_setup(guest: str, env: dict[str, str]) -> str:
    """setup.sh's steps once (no retry) as a bash script, with an ERR trap on post_install.sh."""
    exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    trap = f's=$?; echo "{FAILED_MARK} line $LINENO exit $s: $BASH_COMMAND" >&2'
    return f"""if [ -d {guest}/files ]; then cp -R {guest}/files/. /; fi
export HOME=/root {exports}
cd /
bash -E -c {shlex.quote(f"trap {shlex.quote(trap)} ERR; . {guest}/post_install.sh")}
post=$?
{TMUX_STEP} >/dev/null 2>&1
echo "post_install_exit=$post tmux_exit=$?" >&2
"""


def scan_task(task: Path, template: Template) -> dict:
    container = f"{BUILDER}-{task.name}"
    try:
        docker("run", "-d", "--name", container, f"{BUILDER}:{template.value.lower()}", "sleep", "infinity")
        docker("cp", str(task / "task_setup"), f"{container}:{GUEST}")
        result = docker(
            "exec",
            "-i",
            container,
            "bash",
            "-s",
            check=False,
            timeout=SETUP_TIMEOUT,
            stdin=traced_setup(GUEST, BuildSteps(template).env),
        )
    finally:
        docker("rm", "-f", container, check=False)
    return scan_row(task.name, template, result.stderr)


def scan_row(task: str, template: Template, stderr: str) -> dict:
    failed = [
        {"line": int(match[1]), "exit": int(match[2]), "command": match[3]}
        for match in map(FAILED_LINE.search, stderr.splitlines())
        if match
    ]
    exits = dict(re.findall(r"(post_install_exit|tmux_exit)=(\d+)", stderr))
    return {
        "task": task,
        "template": template.value,
        "failed_steps": failed,
        "post_install_exit": int(exits["post_install_exit"]),
        "tmux_exit": int(exits["tmux_exit"]),
        "stderr_tail": stderr[-3000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shared", type=Path, required=True, help="The shared-image pilot directory.")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file for the results.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = json.loads((args.shared.parent / f"{args.shared.name}.manifest.json").read_text())
    tasks = {task: Template(row["template"]) for task, row in sorted(manifest["tasks"].items())}
    contexts = {
        template: next(args.shared / task / "environment" for task, t in tasks.items() if t is template)
        for template in Template
    }
    had_buildkit = docker("image", "inspect", BUILDKIT_IMAGE, check=False).returncode == 0
    docker("buildx", "create", "--name", BUILDER, "--driver", "docker-container")
    try:
        for template, context in contexts.items():
            docker(
                "buildx",
                "build",
                "--builder",
                BUILDER,
                "--load",
                "-t",
                f"{BUILDER}:{template.value.lower()}",
                str(context),
            )
        with args.output.open("x") as out, ThreadPoolExecutor(args.workers) as pool:
            for row in pool.map(lambda item: scan_task(args.shared / item[0], item[1]), tasks.items()):
                logger.info("%s: %d failed steps", row["task"], len(row["failed_steps"]))
                out.write(json.dumps(row) + "\n")
                out.flush()
    finally:
        docker("rmi", "-f", *(f"{BUILDER}:{template.value.lower()}" for template in Template), check=False)
        docker("buildx", "rm", "--force", BUILDER, check=False)
        if not had_buildkit:
            docker("rmi", "-f", BUILDKIT_IMAGE, check=False)


if __name__ == "__main__":
    main()
