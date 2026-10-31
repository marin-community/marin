# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild wheel-only verifier dependencies and repair the frozen dataset's runtime bindings."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from rolloutengine.models import TaskRolloutSpec

from experiments.evaluation.mini_policy_runtime.environment_inventory import publish
from experiments.evaluation.mini_policy_runtime.profile import DATASET, POLICY_INDEX, REVISION

SOURCE_REQUIREMENTS = (
    "rouge-score==0.1.2",
    "sqlitedict==2.1.0",
    "word2number==1.1",
    "antlr4-python3-runtime==4.11.0",
    "lm-eval @ git+https://github.com/EleutherAI/lm-evaluation-harness@6d642546f4688648fced259eb3302efd36ece5af",
)
ASSET_PATH = "runtime-v3/wheels"


def retarget_evalchemy(output: Path) -> None:
    """Keep evaluator code identical while replacing its Git dependency with the pinned wheel."""
    parquet = hf_hub_download(DATASET, "data/validation.parquet", repo_type="dataset", revision=REVISION)
    row = next(row for row in pq.read_table(parquet).to_pylist() if row["benchmark"] == "math500")
    rollout = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
    requirements = rollout.task.grader.environment.package_spec.pypi_packages
    source = next(Requirement(value).url for value in requirements if Requirement(value).name == "evalchemy")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        original = root / source.split("/")[-1].split("#")[0]
        urllib.request.urlretrieve(source.split("#")[0], original)
        expected = source.split("#sha256=")[1]
        if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
            raise ValueError("Original evaluator wheel hash mismatch")
        subprocess.run(
            [sys.executable, "-m", "wheel", "unpack", str(original), "-d", str(root / "unpacked")], check=True
        )
        package = next((root / "unpacked").iterdir())
        metadata = next(package.glob("*.dist-info/METADATA"))
        original_text = metadata.read_text()
        old = "Requires-Dist: lm-eval[api,math] @ git+https://github.com/EleutherAI/lm-evaluation-harness@v0.4.12"
        if original_text.count(old) != 1:
            raise ValueError("Unexpected evaluator dependency metadata")
        metadata.write_text(original_text.replace(old, "Requires-Dist: lm-eval[api,math]==0.4.12"))
        subprocess.run([sys.executable, "-m", "wheel", "pack", str(package), "-d", str(output)], check=True)


def build_wheels():
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(output), *SOURCE_REQUIREMENTS],
            check=True,
        )
        retarget_evalchemy(output)
        manifest = [
            {"file": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
            for wheel in sorted(output.glob("*.whl"))
        ]
        if len(manifest) != len(SOURCE_REQUIREMENTS) + 1:
            raise ValueError("Dependency wheel build did not produce the expected wheel set")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        commit = HfApi().upload_folder(
            repo_id=POLICY_INDEX,
            repo_type="dataset",
            folder_path=output,
            path_in_repo=ASSET_PATH,
            commit_message="Build source-pinned grading dependencies as wheels for local sandbox execution",
        )
        publish(
            "grading-wheels.json", {"repo": POLICY_INDEX, "revision": commit.oid, "path": ASSET_PATH, "files": manifest}
        )


def repaired_row(row: dict, wheel_requirements: dict[str, str]) -> dict:
    rollout = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
    if rollout.verifier_machine is None or rollout.verifier_machine.backend != "local":
        return row
    environment = rollout.task.grader.environment
    packages = environment.package_spec
    requirements = []
    for requirement in packages.pypi_packages:
        parsed = Requirement(requirement)
        replacement = wheel_requirements.get(canonicalize_name(parsed.name))
        if replacement is None:
            requirements.append(requirement)
        elif parsed.extras:
            requirements.append(replacement.replace(" @ ", "[" + ",".join(sorted(parsed.extras)) + "] @ ", 1))
        else:
            requirements.append(replacement)
    variables = dict(environment.environment_variables)
    if row["benchmark"] == "olympiadbench":
        variables.update({"JUDGE_API_KEY": "${JUDGE_API_KEY}", "JUDGE_BASE_URL": "${JUDGE_BASE_URL}"})
    environment = environment.model_copy(
        update={"package_spec": replace(packages, pypi_packages=tuple(requirements)), "environment_variables": variables}
    )
    task = rollout.task.model_copy(
        update={"grader": rollout.task.grader.model_copy(update={"environment": environment})}
    )
    rollout = rollout.model_copy(
        update={
            "task": task,
            "verifier_machine": rollout.verifier_machine.model_copy(
                update={"cpus": None, "memory_mb": None, "storage_mb": None}
            ),
        }
    )
    TaskRolloutSpec.model_validate_json(rollout.model_dump_json())
    return {**row, "task_spec": task.model_dump_json(), "task_rollout_spec": rollout.model_dump_json()}


def wheel_requirements(wheels_revision: str) -> dict[str, str]:
    manifest_path = hf_hub_download(
        POLICY_INDEX, ASSET_PATH + "/manifest.json", repo_type="dataset", revision=wheels_revision
    )
    requirements = {}
    for item in json.loads(Path(manifest_path).read_text()):
        name = str(parse_wheel_filename(item["file"])[0])
        requirements[name] = (
            name
            + " @ https://huggingface.co/datasets/"
            + POLICY_INDEX
            + "/resolve/"
            + wheels_revision
            + "/"
            + ASSET_PATH
            + "/"
            + item["file"]
            + "#sha256="
            + item["sha256"]
        )
    return requirements


def repaired_table(table: pa.Table, wheels_revision: str) -> pa.Table:
    requirements = wheel_requirements(wheels_revision)
    return pa.Table.from_pylist([repaired_row(row, requirements) for row in table.to_pylist()], schema=table.schema)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-wheels", action="store_true")
    args = parser.parse_args()
    if not args.build_wheels:
        parser.error("Select --build-wheels; dataset repair is called by stage_inputs")
    build_wheels()
