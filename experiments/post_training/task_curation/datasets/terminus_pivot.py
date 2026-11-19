# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA terminal decision points with their declared NeMo string-only scorer."""

import json
from pathlib import Path

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.nemotron_ultra import blend_task, text_request
from taskcompendium.convert.script_grader import grade_script, script_package, shipped_files
from taskcompendium.models import ResourceGroups, TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, NormalizedTask, OracleCommand, RawRow
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.environments import TERMINUS_GRADER_PACKAGES
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

HERE = Path(__file__).parent
DATASET = "nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1"
REVISION = "eaef26944643644c8a3dbbf361ce6128142f5976"
NEMO_REVISION = "951ee9ec835dde26c32302dcb86bcf0ed445bf4d"
VERSION = "2026.11.15.2"
AGENT = "terminus_judge_string_only_simple_agent"
FILES = grade_script(HERE / "terminus_grade.py", *shipped_files(HERE, "terminus_schemas.py"))


def original_answer(_task: TaskSpec) -> OracleCommand:
    return OracleCommand(
        command="mkdir -p /app && cat /solution/reference.json > /app/answer.txt",
        answer_file="/app/answer.txt",
    )


CONTROLS = Controls(golden=original_answer)


def convert(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    request = text_request(row.data, (AGENT,))
    if isinstance(request, ImportRejection):
        return request
    harness = row.data["metadata"]["harness"]
    if harness not in ("terminus_1", "terminus_2"):
        return unsupported("unsupported_terminus_harness", harness)
    try:
        expected = json.loads(row.data["expected_answer"])
    except json.JSONDecodeError as error:
        return source_defect("invalid_expected_action", str(error))
    if not isinstance(expected, dict):
        return source_defect("invalid_expected_action", "Expected action is not a JSON object")
    package = script_package(
        FILES,
        {"expected": expected, "harness": harness, "threshold": row.data.get("threshold", 0.9)},
        environment=required_grader_environment(context),
        timeout=600.0,
        answer_path="/app/answer.txt",
    )
    result = blend_task(row, request, package)
    task = result.task.model_copy(
        update={
            "resources": ResourceGroups(
                verifier=package.resources,
                oracle=(inline_resource("solution/reference.json", row.data["expected_answer"].encode()),),
            ),
            "labels": {"nemo_gym_revision": NEMO_REVISION, "nemo_agent": AGENT},
        }
    )
    return NormalizedTask(task, result.changes)


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=SourceOrigin.NEMOTRON,
                upstream_dataset=DATASET,
                upstream_name="Nemotron-RL-Agentic-Terminal-Pivot-v1",
                build_path="NeMo terminal pivot intake",
                family="terminal-action",
                tags=("license:cc-by-4.0",),
                task_format=TaskFormat.TASKCOMPENDIUM,
                notes=(
                    "Frozen next-action replies with the native string-only NeMo scorer. "
                    "No terminal environment is executed. The 0.9 keystroke-similarity reward can reject valid "
                    "alternative command strings; adding a harmless shell comment to gold commands changes "
                    "reward from 1 to 0. Evidence: s3://marin-rno2a/marin/rl-data/atlas/ingest-a/"
                    "2026.11.15.1/pivot-weak-grader-v1/evidence.json."
                ),
            ),
            config=CurationRecipe(
                name="nemotron-terminal-pivot",
                version=VERSION,
                source=HfSource(DATASET, REVISION, ("atcb_terminal_pivot_release_final_v2.jsonl",), SourceFormat.JSONL),
                convert=convert,
                intended_use=IntendedUse.TRAIN,
                grader=TERMINUS_GRADER_PACKAGES,
                controls=CONTROLS,
                oracle_solutions=True,
                rubric="""Inspect the visible terminal history and requested next action against the private reference.
Check whether the context supports the action, preserves the original task, and avoids exposing future turns.
The native grader checks JSON schema, required completion, and concatenated keystroke similarity at 0.9.
Flag valid alternate actions that this reference comparison would reject and reference actions that are wrong.
These are frozen reply tasks; no terminal actor, live tools or workspace is required.""",
            ),
            pipeline=process_rows,
        )
    ]
