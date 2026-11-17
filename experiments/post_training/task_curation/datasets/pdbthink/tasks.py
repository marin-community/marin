# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PDBThink training tasks with their original tool-free protocol and verifier."""

import base64
import json
import tomllib
from pathlib import Path

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import evidence_resource
from taskcompendium.harbor.execution import native_execution_limits
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import Controls, IntendedUse, OracleCommand, RawRow
from taskcompendium.request_budget import tool_free_prompt_sha256
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import EmptyOutputPolicy

from experiments.post_training.task_curation.datasets.native_harbor import unpack_native
from experiments.post_training.task_curation.environment_fidelity import archive_environment
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin

DATASET = "open-athena/pdbthink-coordinate-tasks"
REVISION = "3734406cb97b1702844319f9a5d860cbbf8fe660"
VERSION = "2026.11.16.1"
BASE = "python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e"
RUBRIC = """
Assess whether the complete coordinate text and question define a valid answer, independently of the released
gold contract. The original verifier scores exact correctness with its released per-family tolerances; partial
credit is diagnostic only. Preserve chain/residue/atom identifiers, displayed precision, rotations, alternate
states and requested FINAL formats. A truncated response, provider refusal or any tool event receives zero
even if its answer text is correct. The model receives only the source system/user prompt and cannot access
tools, files, web, gold contracts or original solutions. Native input/output budgets are model-specific;
released cl100k counts are reference metadata only. Training uses the original train split; validation and
test remain held out. Check missing coordinates, ambiguous selection and inconsistencies between the
question's requested answer format and the original verifier's accepted format without repairing the source.
"""
WRAPPER = Path(__file__).with_name("grade.py")


def decode(row: dict, context: ConversionContext) -> dict:
    if row["split"] != "train":
        raise ValueError("PDBThink training cannot contain held-out rows")
    return {
        **unpack_native(row, context),
        "metadata": {key: value for key, value in row.items() if key not in {"task_binary", "solution_binary"}},
    }


def original_solution(_task: TaskSpec) -> OracleCommand:
    return OracleCommand("/bin/sh /solution/solve.sh", answer_file="/app/answer.txt")


def convert(row: RawRow, _context: ConversionContext) -> TaskSpec:
    data = row.data
    if data["archive_links"]:
        raise ValueError("PDBThink source packages must contain regular files only")
    files = {name: base64.b64decode(value, validate=True) for name, value in data["files"].items()}
    prompt = json.loads(files["prompt.json"])
    if set(prompt) != {"system_prompt", "user_prompt"}:
        raise ValueError("PDBThink public prompt has unexpected fields")
    messages = (
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": prompt["user_prompt"]},
    )
    if tool_free_prompt_sha256(messages) != data["metadata"]["prompt_sha256"]:
        raise ValueError("PDBThink prompt differs from its original row hash")
    config = tomllib.loads(files["task.toml"].decode())
    if config["metadata"]["requires_tool_free_runner"] is not True or config["metadata"]["reward"] != "exact_correct":
        raise ValueError("PDBThink package does not declare its native tool-free scoring contract")
    if files["instruction.md"].decode() != prompt["system_prompt"] + "\n\n" + prompt["user_prompt"]:
        raise ValueError("PDBThink archived instruction differs from public prompt.json")
    if config["environment"]["network_mode"] != "no-network":
        raise ValueError("PDBThink native verifier requires its original no-network environment")
    if "solution/solve.sh" not in files or "solution/answer.txt" not in files:
        raise ValueError("PDBThink original oracle is missing")
    verifier, oracle = [], []
    for name, content in sorted(files.items()):
        target = (
            name.removeprefix("tests/")
            if name.startswith("tests/")
            else (name if name.startswith("solution/") else "source_archive/" + name)
        )
        resource = inline_resource(target, content).model_copy(update=data["file_metadata"][name])
        (oracle if name.startswith("solution/") else verifier).append(resource)
    verifier.append(inline_resource("adapter.py", WRAPPER.read_bytes()))
    verifier.append(
        evidence_resource(
            {
                "metadata": data["metadata"],
                "archive_sha256": data["archive_sha256"],
                "solution_archive_sha256": data["solution_archive_sha256"],
            }
        )
    )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=tuple(TextMessage(**message) for message in messages)),
        environment_requirements=EnvironmentRequirements(),
        execution_requirements=native_execution_limits(config).requirements,
        completion_protocol="tool_free_single",
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("python", "/tests/adapter.py"),
            cwd="/app",
            environment=archive_environment(data["files"], bases={BASE: BASE}),
            answer_path=None,
            reply_status_path="/app/answer_status.json",
            timeout=float(config["verifier"]["timeout_sec"]),
            reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),)),
            empty_output=EmptyOutputPolicy.GRADE,
        ),
        resources=ResourceGroups(verifier=tuple(verifier), oracle=tuple(oracle)),
        tags=("protein-coordinate", "tool-free", "train"),
        labels={
            "family": data["metadata"]["family"],
            "source_group": data["metadata"]["source_group"],
            "split": "train",
        },
    )


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=SourceOrigin.OPENATHENA,
                upstream_dataset="pdbthink-coordinate-tasks",
                upstream_name="PDBThink v1.3.0 coordinate tasks",
                upstream_url="https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks",
                build_path="Original tool-free coordinate tasks",
                family="protein-coordinate",
                tags=("single-turn", "tool-free", "license:apache-2.0"),
                notes=(
                    "Original train split only; exact native budgets and provider status required; "
                    "original verifiers and oracles preserved."
                ),
            ),
            config=CurationRecipe(
                name="pdbthink-coordinate-train",
                source=HfSource(DATASET, REVISION, ("data/train-*.parquet",), SourceFormat.PARQUET, decode=decode),
                convert=convert,
                version=VERSION,
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                controls=Controls(golden=original_solution),
                oracle_solutions=True,
            ),
        )
    ]
