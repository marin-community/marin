# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the standalone QA-abstention and formatting source contracts."""

import json
import re
from dataclasses import replace
from pathlib import Path

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.nemotron_ultra import BlendRequest, text_request
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.grader import GraderPackage
from taskcompendium.models import FileReward, RewardFile, RewardFileFormat, ScriptGrader, TaskSpec
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
GYM_REVISION = "e597c962af405a9c2c6aa91296be51fdcc2f44bb"
GRADER_IMAGE = (
    "ghcr.io/marin-community/iris-task@sha256:c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab"
)
VERSION = "standalone-nemotron-qa-formatting-native-v1"
SOURCES = {
    "abstention": (
        "Nemotron-RL-QA-Abstention-v1",
        "7c2ead1dcfd549874c106c0da8a24554cc4dc746",
        "data/train.jsonl",
        3150,
    ),
    "freeform": (
        "Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1",
        "cc50bed77acc58bd710499402c8903c873db3577",
        "ds2_freeform_train.jsonl",
        9037,
    ),
    "citation": (
        "Nemotron-RL-Instruction-Following-Citation-Formatting-v1",
        "06ef53e2f2ddbac6b7c45dba9f522e3ce3b02584",
        "ds3_citation_train.jsonl",
        9540,
    ),
}
NATIVE_MODULE_PATH = "experiments/post_training/task_curation/datasets"
FORMAT_MODULE = "nemotron_ultra/scorers/skyrl_gym/envs/nemotron_ultra/format_verification.py"
SCORER_PACKAGE_PATHS = (
    "experiments",
    "experiments/post_training",
    "experiments/post_training/task_curation",
    NATIVE_MODULE_PATH,
    f"{NATIVE_MODULE_PATH}/nemotron_ultra",
    f"{NATIVE_MODULE_PATH}/nemotron_ultra/scorers",
    f"{NATIVE_MODULE_PATH}/nemotron_ultra/scorers/skyrl_gym",
    f"{NATIVE_MODULE_PATH}/nemotron_ultra/scorers/skyrl_gym/envs",
    f"{NATIVE_MODULE_PATH}/nemotron_ultra/scorers/skyrl_gym/envs/nemotron_ultra",
)


def convert_native(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    native = dict(row.data)
    request = text_request(
        native, ("abstention_simple_agent", "freeform_formatting_simple_agent", "citation_format_simple_agent")
    )
    if not isinstance(request, BlendRequest):
        return request
    if native.get("tools", []) or (
        "messages" in native and native["messages"] != native["responses_create_params"]["input"]
    ):
        return source_defect("inconsistent_native_request", "Native message copies must agree and tools must be empty")
    config = {"family": "formatting", "gym_revision": GYM_REVISION}
    judge = None
    if request.agent == "abstention_simple_agent":
        question, answer = native.get("question"), native.get("answer")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            return source_defect("missing_abstention_target", "Native semantic QA requires its question and target")
        config.update(family="abstention", question=question, answer=answer)
        judge = replace(
            JUDGE_TIERS[JudgeTier.MEDIUM].spec,
            question=question,
            references=(answer,),
            temperature=0.0,
            max_completion_tokens=64,
        )
    else:
        verifier = native.get("verifier")
        if not isinstance(verifier, dict) or verifier.get("type") not in {"regex", "inline_prose", "string_match"}:
            return unsupported(
                "unhandled_native_format_verifier", "Native verifier must name an implemented format rule"
            )
        pattern_field = "verify_regex" if verifier["type"] in {"regex", "inline_prose"} else "patterns"
        patterns = verifier.get(pattern_field, [])
        if not isinstance(patterns, list) or not all(isinstance(pattern, str) for pattern in patterns):
            return source_defect("invalid_native_patterns", "Native regex patterns must be a list of strings")
        try:
            for pattern in patterns:
                re.compile(pattern)
        except re.error as error:
            return source_defect("invalid_native_regex", str(error))
        config["verifier"] = verifier
    grader = ScriptGrader(
        argv=("python3", "/tests/grade.py"),
        cwd="/tests",
        environment=required_grader_environment(context),
        timeout=None,
        reward=FileReward(files=(RewardFile(path="/app/verdict.json", format=RewardFileFormat.VERDICT_JSON),)),
        judge=judge,
        empty_output=EmptyOutputPolicy.GRADE,
    )
    resources = (
        # Explicit packages keep an installed experiments package from shadowing this bundle.
        *(inline_resource(f"{path}/__init__.py", b"") for path in SCORER_PACKAGE_PATHS),
        inline_resource("grade.py", (HERE / "qa_formatting_grade.py").read_bytes()),
        inline_resource(
            f"{NATIVE_MODULE_PATH}/qa_abstention_native.py", (HERE / "qa_abstention_native.py").read_bytes()
        ),
        inline_resource(f"{NATIVE_MODULE_PATH}/{FORMAT_MODULE}", (HERE / FORMAT_MODULE).read_bytes()),
        inline_resource(
            f"{NATIVE_MODULE_PATH}/qa_abstention_prompt.txt", (HERE / "qa_abstention_prompt.txt").read_bytes()
        ),
        inline_resource("config.json", json.dumps(config, ensure_ascii=False).encode()),
        inline_resource("source-row.json", json.dumps(native, ensure_ascii=False).encode()),
        inline_resource("GYM-LICENSE", (HERE / "qa_formatting_gym_LICENSE").read_bytes()),
    )
    task = conversation_task(row, events=request.context.events, package=GraderPackage(grader, resources))
    return task.model_copy(update={"tags": ("train", "native-qa-formatting"), "labels": {"gym_revision": GYM_REVISION}})


def source(family: str) -> RlDataSource[CurationRecipe]:
    name, revision, filename, count = SOURCES[family]
    return RlDataSource(
        info=SourceInfo(
            origin=SourceOrigin.NVIDIA,
            upstream_dataset=name,
            upstream_name=name,
            build_path=VERSION,
            data_restriction=DataRestriction.OPEN,
            task_format=TaskFormat.TASKSPEC,
            count=count,
            notes=(
                "Original source messages and private verifier metadata are preserved. "
                "Native Gym scoring is pinned at e597c962: formatting counts matching lines or exact citation "
                "markers; abstention extracts the last boxed answer and uses native A/B/C rewards 1/0/0.5. "
                "The undefined genrm_model alias is bound to MEDIUM for semantic abstention judging, retaining "
                "the native prompt, temperature 0, top_p 1 and 64-token cap. No exact-only replacement, "
                "agent budget or native oracle is invented. Conversion does not claim live judge execution."
            ),
        ),
        config=CurationRecipe(
            name="nemotron-standalone-" + family,
            source=HfSource("nvidia/" + name, revision, (filename,), SourceFormat.JSONL),
            convert=convert_native,
            version=VERSION,
            intended_use=IntendedUse.TRAIN,
            grader=Environment(image=GRADER_IMAGE),
        ),
        pipeline=process_rows,
    )


def sources() -> list[RlDataSource[CurationRecipe]]:
    """The declared standalone Nemotron QA abstention and formatting cohorts."""
    return [source(family) for family in sorted(SOURCES)]
