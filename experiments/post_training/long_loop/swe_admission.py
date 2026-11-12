# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run bug-zero/reference-one controls on every derived SWE task in CPU Docker.

Each task gets a fresh instance of its own image and dependency bootstrap.
Only identical receipts may be reused after interruption. No GPU is requested.
"""

import argparse
import base64
import hashlib
import json
import subprocess
import tempfile
import time
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.long_loop.admission_network import private_network
from experiments.post_training.long_loop.execution_policy import DEFAULT_TASK_TIMEOUT_SECONDS
from experiments.post_training.long_loop.pi_rl_data import SOURCE, image_context_digest
from experiments.post_training.long_loop.rl_dataset_admission import file_digest
from experiments.post_training.long_loop.swe_grading import VERIFIER_FINISH_MARGIN
from experiments.post_training.long_loop.swe_verified_data import (
    CURRENT_GRADER,
    GUEST_RUNTIME,
    RUNTIME,
    PreparedTask,
    prepare_task,
)
from experiments.post_training.task_curation.swesmith_staged import DockerPilot, write_json

REFERENCE_PROGRAM = """
import importlib.util,json,pathlib,subprocess,sys
spec=importlib.util.spec_from_file_location('grading', '/opt/long-loop-swe/grading.py')
m=importlib.util.module_from_spec(spec); sys.modules['grading']=m; spec.loader.exec_module(m)
pins=m.TaskPins(**json.loads((m.TRUST/'pins.json').read_text()))
repo=m.TRUST/'repository.git'
trusted=pathlib.Path('/tests/trusted_test_paths.txt').read_text().splitlines()
changed=m.checked(m.git(repo,'diff','--name-only','-z',pins.bug,pins.reference)).decode().split('\\0')
paths=[p for p in changed if p and not m.protected_path(p,trusted)]
assert paths, 'pinned reference has no allowed source change'
patch=m.checked(m.git(repo,'diff','--binary',pins.bug,pins.reference,'--',*paths))
subprocess.run(['git','-c','safe.directory=/testbed','apply','--binary','--whitespace=nowarn','-'],cwd='/testbed',input=patch,check=True)
"""


def materialize(task: TaskSpec, directory: Path) -> None:
    """Stage the existing inline package without changing its paths or bytes."""
    for resource in task.resources.verifier:
        target = directory / resource.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resource_bytes(resource))


def build_image(prepared: PreparedTask, output: Path, *, network: str) -> str:
    digest = image_context_digest(prepared.task)
    directory = output / "images" / digest
    directory.mkdir(parents=True, exist_ok=True)
    materialize(prepared.task, directory)
    pilot = DockerPilot(directory)
    completed = pilot.command(
        [
            "docker",
            "build",
            "--progress=plain",
            "--iidfile",
            str(directory / "image.id"),
            "--platform",
            "linux/amd64",
            "--network",
            network,
            "-f",
            str(directory / "environment/Dockerfile"),
            str(directory / "environment"),
        ],
        timeout=DEFAULT_TASK_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Docker image build failed: {completed.stderr.decode(errors='replace')[-16000:]}")
    image = (directory / "image.id").read_text().strip()
    write_json(directory / "image.json", {"context_sha256": digest, "image": image, "platform": "linux/amd64"})
    return image


def run_controls(prepared: PreparedTask, image: str, directory: Path, *, network: str) -> dict:
    """Use the production bootstrap and grader for both controls, retaining failures."""
    directory.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(prepared.task.model_dump_json().encode()).hexdigest()
    receipt_path = directory / "receipt.json"
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        if prior["task_sha256"] != identity or prior["image"] != image:
            raise ValueError(f"Admission identity changed at {directory}")
        return prior
    pilot = DockerPilot(directory)
    started = time.time()
    result = {
        "task_id": prepared.task.id,
        "task_sha256": identity,
        "pins": asdict(prepared.pins),
        "image": image,
        "network": network,
        "started_unix": started,
        "status": "excluded",
    }
    container = None
    try:
        with tempfile.TemporaryDirectory() as scratch, ExitStack() as lifecycle:
            package = Path(scratch)
            materialize(prepared.task, package)
            config = tomllib.loads((package / "task.toml").read_text())
            resources = config["environment"]
            args = [
                "docker",
                "run",
                "-d",
                "--platform",
                "linux/amd64",
                "--network",
                "none" if network == "slirp4netns" else network,
                "--cpus",
                str(resources.get("cpus", 1)),
                "--memory",
                f"{resources.get('memory_mb', 1024)}m",
            ]
            for key, value in resources.get("env", {}).items():
                args += ["-e", f"{key}={value}"]
            container = pilot.command([*args, "--entrypoint", "sleep", image, "infinity"]).stdout.decode().strip()
            if network == "slirp4netns":
                lifecycle.enter_context(private_network(container, directory / "network.log"))
            pilot.command(["docker", "cp", str(package / "tests"), f"{container}:/tests"])
            payload = base64.b64encode(json.dumps(asdict(prepared.pins)).encode()).decode()
            pilot.command(
                ["docker", "exec", "-w", "/", container, "python3", "-I", "-S", GUEST_RUNTIME, "bootstrap", payload],
                timeout=prepared.pins.setup_timeout + 120,
            )
            controls = {}
            result["controls"] = controls
            for name in ("bug", "reference"):
                if name == "reference":
                    pilot.command(
                        ["docker", "exec", "-w", "/", container, "python3", "-I", "-S", "-c", REFERENCE_PROGRAM],
                        timeout=120,
                    )
                pilot.command(
                    [
                        "docker",
                        "exec",
                        "-w",
                        "/",
                        container,
                        "python3",
                        "-I",
                        "-S",
                        GUEST_RUNTIME,
                        "grade",
                        "--tests",
                        "/tests",
                        "--logs",
                        f"/logs/{name}",
                    ],
                    timeout=prepared.pins.verifier_timeout + 2 * VERIFIER_FINISH_MARGIN,
                )
                local = directory / name
                pilot.command(["docker", "cp", f"{container}:/logs/{name}", str(local)])
                controls[name] = json.loads((local / "verdict.json").read_text())
            if (
                controls["bug"]["status"] == controls["reference"]["status"] == "scored"
                and controls["bug"]["reward"] == 0
                and controls["reference"]["reward"] == 1
            ):
                result["status"] = "admitted"
            else:
                result["reason"] = "bug_reference_control_mismatch"
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        result["reason"] = type(error).__name__
        result["error"] = str(error)
    finally:
        if container is not None:
            # Preserve partial bootstrap/grade evidence even when no verdict exists.
            partial = directory / "bootstrap"
            partial.mkdir(exist_ok=True)
            for name in (
                "pins.json",
                "dependencies.json",
                "candidate-modules.json",
                "bootstrap.json",
                "control-executables.json",
                "setup.stdout",
                "setup.stderr",
            ):
                pilot.command(
                    ["docker", "cp", f"{container}:/opt/long-loop-trusted/{name}", str(partial / name)], check=False
                )
            pilot.command(["docker", "rm", "-f", container], check=False)
    result["ended_unix"] = time.time()
    result["elapsed"] = result["ended_unix"] - started
    write_json(receipt_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--network", required=True, choices=("host", "bridge", "slirp4netns"))
    parser.add_argument("--setup-timeout", type=int, required=True)
    parser.add_argument("--verifier-timeout", type=int, required=True)
    parser.add_argument("--harbor-verifier-timeout", type=int, required=True)
    parser.add_argument(
        "--task-id",
        action="append",
        help="Exact task IDs for bounded regression diagnosis; omission covers every SWE row",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {
        "input_manifest_sha256": file_digest(args.input / "manifest.json"),
        "runtime_sha256": file_digest(RUNTIME),
        "recipe_sha256": file_digest(Path(__file__).with_name("swe_verified_data.py")),
        "runner_sha256": file_digest(Path(__file__)),
        "timeouts": [args.setup_timeout, args.verifier_timeout, args.harbor_verifier_timeout],
    }
    manifest = json.loads((args.input / "manifest.json").read_text())
    for filename in ("agentic-train.parquet", "validation.parquet"):
        if file_digest(args.input / filename) != manifest["files"][filename]["sha256"]:
            raise ValueError(f"Pinned source file changed: {filename}")
    pin_file = args.output / "identity.json"
    if pin_file.exists() and json.loads(pin_file.read_text()) != identity:
        raise ValueError("Admission code or source changed; use a fresh output version")
    write_json(pin_file, identity)
    prepared = []
    exclusions = []
    for filename in ("agentic-train.parquet", "validation.parquet"):
        for batch in pq.ParquetFile(args.input / filename).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                task = TaskSpec.model_validate_json(row["task_spec"])
                if task.source.dataset != SOURCE or (args.task_id and task.id not in args.task_id):
                    continue
                try:
                    item = prepare_task(
                        task,
                        setup_timeout=args.setup_timeout,
                        verifier_timeout=args.verifier_timeout,
                        harbor_verifier_timeout=args.harbor_verifier_timeout,
                        dependency_constraints="",
                        profile=CURRENT_GRADER,
                    )
                except ValueError as error:
                    exclusions.append({"task_id": task.id, "status": "excluded", "reason": str(error)})
                else:
                    prepared.append(item)
    write_json(args.output / "input-exclusions.json", exclusions)
    images = {}
    for item in prepared:
        context = image_context_digest(item.task)
        if context not in images:
            images[context] = build_image(item, args.output, network=args.network)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_controls,
                item,
                images[image_context_digest(item.task)],
                args.output / "tasks" / hashlib.sha256(item.task.id.encode()).hexdigest(),
                network=args.network,
            )
            for item in prepared
        ]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps({key: result[key] for key in ("task_id", "status", "elapsed")}), flush=True)
    receipts = [json.loads(p.read_text()) for p in (args.output / "tasks").glob("*/receipt.json")]
    write_json(
        args.output / "summary.json",
        {
            "identity": identity,
            "selected": len(prepared) + len(exclusions),
            "counts": dict(Counter(r["status"] for r in receipts + exclusions)),
            "complete": len(receipts) == len(prepared),
            "all_source_rows": not bool(args.task_id),
        },
    )


if __name__ == "__main__":
    main()
