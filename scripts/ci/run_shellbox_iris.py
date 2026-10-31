#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the Shellbox live proof from a trusted Iris driver on the Pod network."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import iris.cluster.types
import shellbox.backends.iris.machine
from iris.cli.connect import open_iris_client
from iris.cluster.client.job_info import get_job_info
from iris.cluster.config import load_config
from iris.cluster.types import Entrypoint, EnvironmentSpec, NetworkAttachment, ResourceSpec
from iris.resources.state import JobState
from iris.rpc import job_pb2
from rigging.timing import Duration

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_TEST = "lib/shellbox/tests/test_iris_service_group_live.py"
SCRIPT = "scripts/ci/run_shellbox_iris.py"
MANIFEST = "_shellbox_ci_sources.json"
BUNDLE_INCLUDES = (LIVE_TEST, "lib/shellbox/tests/conftest.py", "lib/taskcompendium/pyproject.toml", SCRIPT)
PYTEST_COMMAND = [
    "uv",
    "run",
    "--no-sync",
    "--package",
    "marin-shellbox",
    "--extra",
    "iris",
    "--group",
    "test",
    "pytest",
    "-c",
    "lib/taskcompendium/pyproject.toml",
    LIVE_TEST,
    "-m",
    "integration and requires_cluster",
    "-o",
    "addopts=",
    "--timeout=900",
    "-v",
    "-s",
]


def source_manifest() -> dict:
    paths = {REPO_ROOT / path for path in (*BUNDLE_INCLUDES, "pyproject.toml", "uv.lock")}
    for package in ("iris", "shellbox"):
        paths.update((REPO_ROOT / "lib" / package / "src").rglob("*.py"))
    return {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "files": {
            str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)
        },
    }


def run_driver() -> int:
    info = get_job_info()
    if info is None or not info.attempt_uid or not info.task_token:
        raise RuntimeError("The live proof must run inside an authenticated Iris driver attempt")
    manifest = json.loads((REPO_ROOT / MANIFEST).read_text())
    for relative, expected in manifest["files"].items():
        if hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Bundled source differs from submitted checkout: {relative}")
    # Editable workspace imports must win over any image-provided installation.
    for module, relative in (
        (iris.cluster.types, "lib/iris/src/iris/cluster/types.py"),
        (shellbox.backends.iris.machine, "lib/shellbox/src/shellbox/backends/iris/machine.py"),
    ):
        if Path(module.__file__).resolve() != REPO_ROOT / relative:
            raise RuntimeError(f"Live proof imported an image-installed module: {module.__name__}")
    print(
        f"Shellbox live proof: revision={manifest['revision']} driver={info.task_id} attempt={info.attempt_uid}",
        flush=True,
    )
    return subprocess.run(PYTEST_COMMAND, cwd=REPO_ROOT, check=False).returncode


def submit_driver(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    environment = EnvironmentSpec(
        env_vars={
            "SHELLBOX_IRIS_CONTROLLER_URL": config.kubernetes_provider.controller_address,
            "SHELLBOX_IRIS_PYTHON_IMAGE": args.python_image,
            "SHELLBOX_IRIS_PUBLIC_ADDRESS": args.public_address,
            "SHELLBOX_IRIS_PRIVATE_ADDRESS": args.private_address,
        },
        setup_scripts=[
            'set -e\ncd "$IRIS_WORKDIR"\n' "uv sync --frozen --package marin-shellbox --extra iris --group test\n"
        ],
    )
    entrypoint = Entrypoint(
        command=[
            "uv",
            "run",
            "--no-sync",
            "--package",
            "marin-shellbox",
            "--extra",
            "iris",
            "--group",
            "test",
            "python",
            SCRIPT,
            "driver",
        ],
        workdir_files={MANIFEST: json.dumps(source_manifest()).encode()},
    )
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    with open_iris_client(
        controller_url=args.controller_url, workspace=REPO_ROOT, extra_bundle_includes=BUNDLE_INCLUDES
    ) as client:
        job = client.submit(
            entrypoint,
            name=args.job_name,
            resources=ResourceSpec(cpu=2, memory="4GB", disk="8GB"),
            environment=environment,
            network_attachment=NetworkAttachment.POD,
            egress_policy=job_pb2.EGRESS_POLICY_CLUSTER,
            max_retries_failure=0,
            max_retries_preemption=0,
            scheduling_timeout=Duration.from_seconds(180),
            timeout=Duration.from_seconds(1200),
        )
        print(f"Shellbox driver job: {job.job_id.to_wire()}", flush=True)
        try:
            if github_env := os.environ.get("GITHUB_ENV"):
                with Path(github_env).open("a") as output:
                    output.write(f"SHELLBOX_IRIS_DRIVER_JOB={job.job_id.to_wire()}\n")
            status = job.wait(timeout=1320, poll_interval=5, raise_on_failure=False)
            return 0 if status.state == JobState.SUCCEEDED else 1
        finally:
            try:
                lines = [entry.data for entry in job.logs(max_lines=2000, tail=True)]
                args.log_path.write_text("\n".join(lines) + "\n")
                for line in lines:
                    print(line)
            finally:
                job.cancel()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit", help="Bundle this checkout and wait for the in-cluster proof")
    submit.add_argument("--controller-url", required=True)
    submit.add_argument("--config", type=Path, required=True)
    submit.add_argument("--job-name", required=True)
    submit.add_argument("--python-image", required=True)
    submit.add_argument("--public-address", required=True)
    submit.add_argument("--private-address", required=True)
    submit.add_argument("--log-path", type=Path, required=True)
    commands.add_parser("driver", help="Run the bundled test inside the submitted driver")
    args = parser.parse_args()
    return submit_driver(args) if args.command == "submit" else run_driver()


if __name__ == "__main__":
    sys.exit(main())
