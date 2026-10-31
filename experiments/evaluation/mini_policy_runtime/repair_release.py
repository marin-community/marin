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

from experiments.evaluation.mini_policy_runtime.artifacts import publish
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


def export_repaired(output: Path, wheels_revision: str) -> None:
    """Rebuild both releases with identical task membership, prompts and repetitions."""
    releases = (
        ("mini", DATASET, REVISION, 840, 1626),
        ("micro", "open-athena/Marin-Micro-Eval-210", "4564ed081c316bc3231ae5c3a53a46b39248c819", 210, 450),
    )
    receipts = []
    for name, dataset, revision, unique_count, rollout_count in releases:
        for filename, expected in (("validation.parquet", unique_count), ("validation-rollouts.parquet", rollout_count)):
            source = hf_hub_download(dataset, "data/" + filename, repo_type="dataset", revision=revision)
            original = pq.read_table(source)
            repaired = repaired_table(original, wheels_revision)
            if repaired.num_rows != expected:
                raise ValueError(f"Unexpected {dataset} row count")
            for before, after in zip(original.to_pylist(), repaired.to_pylist(), strict=True):
                before_task = TaskRolloutSpec.model_validate_json(before["task_rollout_spec"]).task
                after_task = TaskRolloutSpec.model_validate_json(after["task_rollout_spec"]).task
                if before_task.model_copy(update={"grader": after_task.grader}) != after_task:
                    raise ValueError(f"Repair changed task content: {before['task_id']}")
                if any(before[key] != after[key] for key in before if key not in ("task_spec", "task_rollout_spec")):
                    raise ValueError(f"Repair changed row membership or provenance: {before['task_id']}")
            destination = output / name / "data" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(repaired, destination)
            receipts.append(
                {
                    "dataset": dataset,
                    "source_revision": revision,
                    "file": str(destination.relative_to(output)),
                    "rows": expected,
                    "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                }
            )
    (output / "repair-manifest.json").write_text(
        json.dumps({"wheels_revision": wheels_revision, "files": receipts}, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build-wheels", action="store_true")
    group.add_argument("--output", type=Path)
    parser.add_argument("--wheels-revision")
    args = parser.parse_args()
    if args.build_wheels:
        build_wheels()
    elif args.wheels_revision:
        export_repaired(args.output, args.wheels_revision)
    else:
        parser.error("Dataset export requires --wheels-revision")
