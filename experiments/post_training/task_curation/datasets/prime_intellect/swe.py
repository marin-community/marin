# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned verified SWE rows packaged behind their original image and scoring boundary."""

import base64
import hashlib
import json
import shlex
from dataclasses import replace
from pathlib import Path

import tomlkit
from taskcompendium.convert.answers import source_defect
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow

from experiments.post_training.task_curation.datasets.native_harbor import native_task
from experiments.post_training.task_curation.datasets.prime_intellect.native_scripts import (
    ENV,
    INSTALL_RG,
    LINK_VENV,
    SOLVER_SYSTEM_PROMPT,
    _ids,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

HERE = Path(__file__).parent
VERSION = "2026.11.15.3"
TASKSET_REVISION = "b41d8759070715dc102000e06fba0d73ebce9955"
RUBRIC = """Compare the public issue with the original hidden tests and golden patch. Check whether the
issue supplies the required behavior and the tests measure it without demanding unrelated changes.
The per-task repository image is part of the contract. Do not mistake a missing local checkout or
unexecuted Docker image for an underspecified issue. Do not expose private tests or the golden patch.
"""


def native_package(
    row: RawRow,
    context: ConversionContext,
    *,
    image: str,
    workdir: str,
    setup: str,
    config: dict,
    grader: str,
    patch_commands: tuple[str, ...],
) -> TaskSpec | ImportRejection:
    gold_patch = row.data["patch"]
    if not gold_patch.strip():
        return source_defect("missing_gold_patch", "The verified source does not provide its original solution patch")
    if not config["test_ids"]:
        return source_defect("missing_test_ids", "The source scorer requires nonempty F2P/P2P test IDs")
    task_config = {
        "environment": {
            "workdir": workdir,
            "cpus": 4,
            "memory_mb": 4096,
            "storage_mb": 10240,
            "allow_internet": True,
            "env": ENV,
            "setup": {"command": setup, "timeout_sec": 1800.0},
        },
        "verifier": {"environment_mode": "shared"},
    }
    oracle = "#!/bin/sh\nset -eu\ncd " + shlex.quote(workdir) + "\n" + " || ".join(patch_commands) + "\n"
    files = {
        "instruction.md": (SOLVER_SYSTEM_PROMPT + "\n\n" + row.data["problem_statement"]).encode(),
        "task.toml": tomlkit.dumps(task_config).encode(),
        "environment/Dockerfile": ("FROM " + image + "\nWORKDIR " + workdir + "\n").encode(),
        "tests/test.sh": b"#!/bin/sh\nset -eu\nmkdir -p /logs/verifier\npython /tests/grade.py\n",
        "tests/grade.py": (HERE / grader).read_bytes(),
        "tests/native_scripts.py": (HERE / "native_scripts.py").read_bytes(),
        "tests/config.json": json.dumps(config, sort_keys=True).encode(),
        "solution/solve.sh": oracle.encode(),
        "solution/gold.patch": gold_patch.encode(),
    }
    if grader == "scale_grade.py":
        files["tests/scale_score.py"] = (HERE / "scale_score.py").read_bytes()
    decoded = {
        "files": {name: base64.b64encode(content).decode() for name, content in files.items()},
        "file_metadata": {name: {"mode": "0755" if name.endswith(".sh") else "0644", "mtime_ns": 0} for name in files},
        "archive_links": {},
        "archive_sha256": hashlib.sha256(json.dumps(row.data, sort_keys=True).encode()).hexdigest(),
    }
    task = native_task(replace(row, data=decoded), context)
    if isinstance(task, ImportRejection):
        return task
    return task.model_copy(update={"labels": {**task.labels, "prime_taskset_revision": TASKSET_REVISION}})


def convert_scale(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    data = row.data
    return native_package(
        row,
        context,
        image=data["image_url"],
        workdir=data["workdir"],
        setup="cd " + shlex.quote(data["workdir"]) + " && " + data["pre_commands"].strip().removesuffix("\\n"),
        config={
            "workdir": data["workdir"],
            "base_commit": data["parent_commit"],
            "f2p_patch": data["f2p_patch"],
            "f2p_script": data["f2p_script"],
            "test_ids": _ids(data["FAIL_TO_PASS"]) + _ids(data["PASS_TO_PASS"]),
        },
        grader="scale_grade.py",
        patch_commands=(
            "git apply --verbose /solution/gold.patch",
            "git apply --verbose --ignore-space-change --ignore-whitespace /solution/gold.patch",
            "patch --batch --fuzz=5 -p1 -i /solution/gold.patch",
            "git apply --verbose --reject --ignore-whitespace /solution/gold.patch",
        ),
    )


def convert_lego(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    data = row.data
    return native_package(
        row,
        context,
        image=data["image_name"],
        workdir="/testbed",
        setup=LINK_VENV + "; " + INSTALL_RG,
        config={
            "base_commit": data["base_commit"],
            "test_patch": data["test_patch"],
            "test_cmd": data["test_cmd"],
            "test_ids": list(data["FAIL_TO_PASS"]) + list(data["PASS_TO_PASS"]),
        },
        grader="lego_grade.py",
        patch_commands=(
            "git apply --whitespace=fix /solution/gold.patch",
            "patch --batch --fuzz=5 -p1 -i /solution/gold.patch",
        ),
    )


def sources() -> list[RlDataSource[CurationRecipe]]:
    definitions = (
        (
            "Scale-SWE-Verified",
            "2c29b67e0ea4d7819f60d8a10d7475a7ad89ccd6",
            "scale-swe-verified",
            "train",
            2,
            convert_scale,
        ),
        (
            "SWE-Lego-Real-Data-Verified",
            "0a1b227ec41f0e96cb8f5c7a1c225258a80edb8d",
            "swe-lego-real-data-verified",
            "resolved",
            3,
            convert_lego,
        ),
    )
    return [
        RlDataSource(
            info=SourceInfo(
                origin=SourceOrigin.PRIME_INTELLECT,
                upstream_dataset=f"PrimeIntellect/{dataset}",
                upstream_name=dataset,
                build_path="Prime Intellect verified SWE intake",
                family="swe-repo",
                tags=("license:cc-by-4.0",) if dataset == "Scale-SWE-Verified" else ("license:apache-2.0",),
                task_format=TaskFormat.HARBOR,
                notes=(
                    f"Original source image and scoring contract; taskset code {TASKSET_REVISION}. "
                    "Pending qualification."
                ),
            ),
            config=CurationRecipe(
                name=name,
                version=VERSION,
                source=HfSource(
                    f"PrimeIntellect/{dataset}",
                    revision,
                    tuple(f"data/{split}-{i:05d}-of-{parts:05d}.parquet" for i in range(parts)),
                    SourceFormat.PARQUET,
                ),
                convert=convert,
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                oracle_solutions=True,
            ),
            pipeline=process_rows,
        )
        for dataset, revision, name, split, parts, convert in definitions
    ]
