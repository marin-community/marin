# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve InverseIFEval's private source checks and VerifIF ALL aggregation."""

import json
from dataclasses import replace
from pathlib import Path

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.grader import GraderPackage
from taskcompendium.models import FileReward, RewardFile, RewardFileFormat, ScriptGrader, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow
from taskcompendium.pipeline.reviewer_spec import JudgeTier
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import EmptyOutputPolicy

from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.judges import JUDGE_TIERS
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

HERE = Path(__file__).parent
REPO = "nvidia/Nemotron-RL-InverseIFEval-v1"
REVISION = "6958020c40dc3df170ed3ef90f88d6c290427148"
SOURCE_ID = "Nvidia:Nemotron-RL-InverseIFEval-v1"
VERSION = "inverse-ifeval-native-checks-v1"
GYM_REVISION = "e597c962af405a9c2c6aa91296be51fdcc2f44bb"
GRADER_IMAGE = (
    "ghcr.io/marin-community/iris-task@sha256:c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab"
)
# The prompt below is from resources_servers/verifif/vif_validators/data_loader.py at GYM_REVISION.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. Apache-2.0; see inverse_ifeval_gym_LICENSE.
JUDGE_TEMPLATE = """
You are an impartial LLM-Judge. Evaluate the model response based strictly on the specific question provided below.

QUESTION TO EVALUATE AGAINST:
{question}

MODEL RESPONSE TO EVALUATE:
{model_response}

Does the model's response satisfy the criterion described in the evaluation question?
Analyze carefully, then respond with exactly [[YES]] or [[NO]] on the last line.
"""


def convert_inverse(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    native = dict(row.data)
    if native.get("instructions") != [] or native.get("tools") != []:
        return unsupported(
            "non_judge_inverse_contract",
            "Only source rows with empty deterministic instructions and tools are supported",
        )
    messages = native.get("responses_create_params", {}).get("input")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or messages[0].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
        or not messages[0]["content"].strip()
        or native.get("messages") != messages
    ):
        return source_defect(
            "invalid_inverse_conversation", "Public message copies must agree on one nonempty user prompt"
        )
    checks = native.get("llm_judge")
    if not isinstance(checks, list) or not checks:
        return source_defect("missing_inverse_checks", "Source must supply nonempty private judge checks")
    identifiers = set()
    for check in checks:
        if (
            not isinstance(check, dict)
            or not isinstance(check.get("uid"), int)
            or check["uid"] in identifiers
            or not isinstance(check.get("content"), str)
            or not check["content"].strip()
            or check.get("pass_criteria") not in {"YES", "NO"}
            or check.get("source") not in {"user", "system"}
            or check.get("is_misalignment_check") is not False
        ):
            return unsupported(
                "unhandled_inverse_check", "Preserve incomplete, duplicate or misalignment checks for typed repair"
            )
        identifiers.add(check["uid"])
    judge = replace(
        JUDGE_TIERS[JudgeTier.MEDIUM].spec,
        references=("Evaluate each original private check independently; all expected verdicts must match.",),
        max_completion_tokens=10000,
    )
    grader = ScriptGrader(
        argv=("python3", "/tests/grade.py"),
        cwd="/",
        environment=required_grader_environment(context),
        timeout=None,
        reward=FileReward(files=(RewardFile(path="/app/verdict.json", format=RewardFileFormat.VERDICT_JSON),)),
        judge=judge,
        empty_output=EmptyOutputPolicy.GRADE,
    )
    resources = (
        inline_resource("grade.py", (HERE / "inverse_ifeval_grade.py").read_bytes()),
        inline_resource(
            "config.json",
            json.dumps(
                {
                    "checks": checks,
                    "native_judge_template": JUDGE_TEMPLATE,
                    "top_p": 0.8,
                    "gym_revision": GYM_REVISION,
                    "aggregation": "all",
                },
                ensure_ascii=False,
            ).encode(),
        ),
        inline_resource("source-row.json", json.dumps(native, ensure_ascii=False).encode()),
        inline_resource("GYM-LICENSE", (HERE / "inverse_ifeval_gym_LICENSE").read_bytes()),
    )
    task = conversation_task(
        row, events=(TextMessage(role="user", content=messages[0]["content"]),), package=GraderPackage(grader, resources)
    )
    return task.model_copy(
        update={
            "tags": ("train", "instruction-following", "native-private-checks"),
            "labels": {"source_id": SOURCE_ID, "native_id": str(native["id"]), "gym_revision": GYM_REVISION},
        }
    )


def source() -> RlDataSource[CurationRecipe]:
    return RlDataSource(
        info=SourceInfo(
            origin=SourceOrigin.NVIDIA,
            upstream_dataset="Nemotron-RL-InverseIFEval-v1",
            upstream_name="Nemotron-RL-InverseIFEval-v1",
            build_path=VERSION,
            data_restriction=DataRestriction.OPEN,
            task_format=TaskFormat.TASKSPEC,
            count=1000,
            notes=(
                "Original checks, prompt, YES/NO polarity, independent calls and ALL reward preserved. "
                "Undefined policy-model judge bound to MEDIUM. Malformed verdicts are infrastructure errors. "
                "Native data declares no agent environment or task budget."
            ),
        ),
        config=CurationRecipe(
            name="nemotron-inverse-ifeval",
            source=HfSource(REPO, REVISION, ("data/train.jsonl",), SourceFormat.JSONL),
            convert=convert_inverse,
            version=VERSION,
            intended_use=IntendedUse.TRAIN,
            grader=Environment(image=GRADER_IMAGE),
        ),
        pipeline=process_rows,
    )
