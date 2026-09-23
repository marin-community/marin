# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run prepared native checks sequentially on an explicitly provisioned CPU worker.

This entry point never provisions infrastructure or makes model calls. Package
installation needs network access; every biological input is already in the bundle.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from experiments.post_training.bio_tasks.native.commands import THREAD_VARIABLES

INSTALL_TIMEOUT = 900
CASE_TIMEOUT = 900
RUN_TIMEOUT = 6 * 60 * 60
MAX_USED_GIB = 40


def run_command(argv: list[str], destination: Path, timeout: int, environment: dict[str, str]) -> int:
    """Retain setup and subprocess failures without conflating them with wrong biology."""
    start = time.monotonic()
    with destination.open("wb") as output:
        try:
            result = subprocess.run(
                argv, stdout=output, stderr=subprocess.STDOUT, env=environment, timeout=timeout, check=False
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(
            {
                "argv": argv,
                "exit_code": code,
                "elapsed_seconds": time.monotonic() - start,
            },
            indent=2,
        )
        + "\n"
    )
    return code


def run(bundle: Path, output: Path, micromamba: Path) -> None:
    bundle, output, micromamba = bundle.resolve(), output.resolve(), micromamba.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((bundle / "plan.json").read_text())
    environment = dict(os.environ)
    environment.update(dict.fromkeys(THREAD_VARIABLES, "1"))
    environment["MAMBA_ROOT_PREFIX"] = str(output / "mamba")
    environment["PYTHONPATH"] = str(bundle / "code")
    environment["PYTHONNOUSERSITE"] = "1"
    prefix = output / "environment"
    start = time.monotonic()
    initial_free = shutil.disk_usage(output).free
    results = []
    try:
        for repository in plan["repositories"]:
            index = repository["repository_index"]
            result = {"repository_index": index, "cases": [], "verification": "pending"}
            results.append(result)
            if repository["adapter"] == "pending":
                result["execution"] = "adapter_pending"
                continue
            if time.monotonic() - start >= RUN_TIMEOUT:
                result["execution"] = "run_time_limit"
                break
            free = shutil.disk_usage(output).free
            if free < 5 * 1024**3 or initial_free - free > MAX_USED_GIB * 1024**3:
                result["execution"] = "disk_limit"
                break
            repo_dir = output / f"{index:02d}"
            repo_dir.mkdir()
            install = [
                str(micromamba),
                "create",
                "-y",
                "--no-rc",
                "--strict-channel-priority",
                "-p",
                str(prefix),
                "-c",
                "conda-forge",
                "-c",
                "bioconda",
                *repository["package_specs"],
            ]
            code = run_command(install, repo_dir / "install.log", INSTALL_TIMEOUT, environment)
            if code:
                result["execution"] = "environment_failed"
                result["install_exit_code"] = code
                if prefix.exists():
                    shutil.rmtree(prefix)
                continue
            run_command(
                [str(micromamba), "list", "-p", str(prefix), "--explicit"],
                repo_dir / "environment.explicit.txt",
                60,
                environment,
            )
            result["execution"] = "completed"
            for case in repository["cases"]:
                target = repo_dir / case["task_id"]
                argv = [
                    str(micromamba),
                    "run",
                    "-p",
                    str(prefix),
                    "python",
                    "-m",
                    "experiments.post_training.bio_tasks.native.check",
                    "--repository",
                    str(index),
                    "--inputs",
                    str(bundle / case["inputs"]),
                    "--output",
                    str(target),
                ]
                code = run_command(argv, repo_dir / (case["task_id"] + ".log"), CASE_TIMEOUT, environment)
                result["cases"].append(
                    {"task_id": case["task_id"], "exit_code": code, "output": str(target.relative_to(output))}
                )
                if code:
                    result["execution"] = "operation_failed"
            shutil.rmtree(prefix)
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        if prefix.exists():
            shutil.rmtree(prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micromamba", type=Path, required=True)
    args = parser.parse_args()
    run(args.bundle, args.output, args.micromamba)


if __name__ == "__main__":
    main()
