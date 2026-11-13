# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check, under local Docker, that a shared-image task reproduces its per-task image.

For each task: build the per-task reference image from the packaged pilot, and start the task's shared image (A or
B) and apply its ``task_setup/`` with Harbor's own ``task_setup_command``, as the trial does. Then compare:

- the filesystem under ``STATE_ROOTS`` (type, mode, owner, size, content hash, link target);
- installed packages (``dpkg-query``, ``pip freeze``) and the image environment;
- the verifier's reward with no agent action (the no-op score).

Builds use a dedicated buildx builder, so removing it removes their cache. Every image, container and the builder
are removed at the end. CPU only; no model or GPU. Runs in the vendored Harbor runtime, from the repository root:

    PYTHONPATH=$PWD uv run --isolated --project vendor/harbor --extra runtime --no-default-groups --frozen \\
        python -m experiments.post_training.tmax_reserve.parity --source ... --shared ... --tasks ... --output ...
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

from harbor.trial.trial import TASK_SETUP_GUEST_PREFIX, task_setup_command

from experiments.post_training.tmax_reserve.shared_image import Template, parse_dockerfile

logger = logging.getLogger(__name__)

BUILDER = "tmax-parity"
# The image a docker-container buildx builder runs.
BUILDKIT_IMAGE = "moby/buildkit:buildx-stable-1"
STATE_ROOTS = ("/home", "/app", "/root", "/opt", "/srv", "/usr/local", "/etc", "/tmp")
# Per-container files and caches that differ between any two containers or builds.
VOLATILE = ("/etc/hostname", "/etc/hosts", "/etc/resolv.conf", "/etc/mtab", "/root/.cache", "/tmp/.harbor-task-setup")
GUEST = f"{TASK_SETUP_GUEST_PREFIX}parity"
STATE_SCRIPT = f"""
import hashlib, json, os, stat, subprocess
roots, volatile = {list(STATE_ROOTS)!r}, {list(VOLATILE)!r}
files = {{}}
for root in roots:
    for directory, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if not any(os.path.join(directory, d).startswith(v) for v in volatile)]
        for name in [*dirs, *names]:
            path = os.path.join(directory, name)
            if any(path.startswith(v) for v in volatile):
                continue
            info = os.lstat(path)
            row = [stat.filemode(info.st_mode), info.st_uid, info.st_gid]
            if stat.S_ISLNK(info.st_mode):
                row.append(os.readlink(path))
            elif stat.S_ISREG(info.st_mode):
                with open(path, "rb") as handle:
                    row += [info.st_size, hashlib.sha256(handle.read()).hexdigest()]
            files[path] = row
def lines(*argv):
    result = subprocess.run(argv, capture_output=True, text=True)
    return sorted(result.stdout.splitlines()) if result.returncode == 0 else ["<exit %d>" % result.returncode]
print(json.dumps({{
    "files": files,
    "dpkg": lines("dpkg-query", "-W", "-f", "${{Package}}=${{Version}}\\n"),
    "pip": lines("python3", "-m", "pip", "list", "--format=freeze"),
}}))
"""


def docker(*argv: str, check: bool = True, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *argv], capture_output=True, text=True, check=check, input=stdin)


def state(container: str) -> dict:
    return json.loads(docker("exec", "-i", container, "python3", "-I", "-", stdin=STATE_SCRIPT).stdout)


def image_env(image: str) -> list[str]:
    return sorted(json.loads(docker("image", "inspect", "--format", "{{json .Config.Env}}", image).stdout))


def noop_reward(container: str, tests: Path) -> str:
    docker("cp", f"{tests}/.", f"{container}:/tests")
    docker("exec", container, "bash", "-c", "mkdir -p /logs/verifier && bash /tests/test.sh", check=False)
    return docker("exec", container, "cat", "/logs/verifier/reward.txt", check=False).stdout.strip()


def _compared(path: str, state: dict) -> list:
    """The fields compared for one path; bytecode embeds its source's mtime, so only its mode and owner count."""
    row = state["files"][path]
    return row[:3] if path.endswith(".pyc") else row


def diff(reference: dict, shared: dict) -> dict:
    """Paths and packages that differ, with a few examples of each."""
    common = set(reference["files"]) & set(shared["files"])
    files = {
        "only_reference": sorted(set(reference["files"]) - set(shared["files"])),
        "only_shared": sorted(set(shared["files"]) - set(reference["files"])),
        "changed": sorted(path for path in common if _compared(path, reference) != _compared(path, shared)),
        "bytecode_content_changed": sorted(
            path for path in common if path.endswith(".pyc") and reference["files"][path] != shared["files"][path]
        ),
    }
    packages = {
        kind: {
            "only_reference": sorted(set(reference[kind]) - set(shared[kind])),
            "only_shared": sorted(set(shared[kind]) - set(reference[kind])),
        }
        for kind in ("dpkg", "pip")
    }
    return {
        "files": {key: {"count": len(paths), "examples": paths[:20]} for key, paths in files.items()},
        "packages": packages,
    }


def check_task(task: str, source: Path, shared: Path, images: dict[Template, str]) -> dict:
    environment = source / task / "environment"
    template = parse_dockerfile((environment / "Dockerfile").read_text()).template
    reference_image = f"{BUILDER}-ref:{task}"
    started = time.monotonic()
    docker("buildx", "build", "--builder", BUILDER, "--load", "-t", reference_image, str(environment))
    build_seconds = time.monotonic() - started
    containers = {"reference": f"{BUILDER}-ref-{task}", "shared": f"{BUILDER}-shared-{task}"}
    try:
        docker("run", "-d", "--name", containers["reference"], reference_image, "sleep", "infinity")
        docker("run", "-d", "--name", containers["shared"], images[template], "sleep", "infinity")
        docker("cp", str(shared / task / "task_setup"), f"{containers['shared']}:{GUEST}")
        started = time.monotonic()
        setup = docker("exec", "-w", "/", containers["shared"], "bash", "-c", task_setup_command(GUEST), check=False)
        setup_seconds = time.monotonic() - started
        docker("exec", containers["shared"], "rm", "-rf", GUEST)
        states = {side: state(container) for side, container in containers.items()}
        rewards = {side: noop_reward(container, source / task / "tests") for side, container in containers.items()}
        env_equal = image_env(reference_image) == image_env(images[template])
    finally:
        docker("rm", "-f", *containers.values(), check=False)
        docker("rmi", "-f", reference_image, check=False)
    return {
        "task": task,
        "template": template.value,
        "setup_exit": setup.returncode,
        "setup_stderr_tail": setup.stderr[-2000:],
        "build_seconds": round(build_seconds),
        "setup_seconds": round(setup_seconds),
        "noop_rewards": rewards,
        "env_equal": env_equal,
        "diff": diff(states["reference"], states["shared"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="The packaged per-task pilot directory.")
    parser.add_argument("--shared", type=Path, required=True, help="The shared-image pilot directory.")
    parser.add_argument("--tasks", required=True, help="Comma-separated task names.")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file for the results.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = json.loads((args.shared.parent / f"{args.shared.name}.manifest.json").read_text())
    images = {Template(template): image for template, image in manifest["images"].items()}
    results = []
    output = {"docker_system_df_before": docker("system", "df").stdout, "results": results}
    # Only images this run pulled are removed afterwards.
    present = {
        image
        for image in [*images.values(), BUILDKIT_IMAGE]
        if docker("image", "inspect", image, check=False).returncode == 0
    }
    docker("buildx", "create", "--name", BUILDER, "--driver", "docker-container")
    try:
        for template_image in images.values():
            docker("pull", template_image)
        for task in args.tasks.split(","):
            logger.info("checking %s", task)
            results.append(check_task(task, args.source, args.shared, images))
            args.output.write_text(json.dumps(output, indent=1))
    finally:
        docker("buildx", "rm", "--force", BUILDER, check=False)
        pulled = [image for image in [*images.values(), BUILDKIT_IMAGE] if image not in present]
        if pulled:
            docker("rmi", "-f", *pulled, check=False)
        output["docker_system_df_after"] = docker("system", "df").stdout
        args.output.write_text(json.dumps(output, indent=1))
    print(json.dumps([{key: row[key] for key in ("task", "setup_exit", "noop_rewards")} for row in results], indent=1))


if __name__ == "__main__":
    main()
