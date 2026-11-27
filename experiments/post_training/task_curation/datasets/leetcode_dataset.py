# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert LeetCodeDataset's train problems with the source's own test harness as the grader."""

from pathlib import Path

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.script_grader import grade_script, script_package
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.models import TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizationChange, NormalizedTask, RawRow
from verifyit.spec import DEFAULT_OUTPUT

from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

HERE = Path(__file__).parent
REPO = "newfacade/LeetCodeDataset"
REVISION = "215604aeed660029df7de2fea5a4d7b6ed476a08"
SOURCE_ID = "Newfacade:LeetCodeDataset"
VERSION = "leetcode-harness-grader-v1"
TRAIN_COUNT = 2641
GRADER_IMAGE = (
    "ghcr.io/marin-community/iris-task@sha256:c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab"
)
GRADER_TIMEOUT = 330.0
COMPLETION_INSTRUCTION = "\nReturn the complete Python solution in this format:\n```python\n# solution\n```"
LCB_GRADE = grade_script(HERE / "leetcode_grade.py")


def convert_problem(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    native = dict(row.data)
    required = {"prompt", "starter_code", "entry_point", "test", "task_id", "difficulty", "query", "response"}
    if not required <= set(native) or not isinstance(native["prompt"], str) or not native["prompt"].strip():
        return source_defect("incomplete_leetcode_contract", "The task fields with a nonempty prompt are required")
    for field in ("starter_code", "entry_point", "test"):
        if not isinstance(native[field], str) or not native[field].strip():
            return source_defect("incomplete_leetcode_contract", f"{field} must be nonempty text")
    callable_name = native["entry_point"].replace("(", " ").split()[-1].split(".")[-1]
    if callable_name not in native["starter_code"]:
        return source_defect("entry_point_not_in_starter", "The entry point must name the starter code's callable")
    if "def check(" not in native["test"]:
        return source_defect("missing_check_harness", "The source test harness must define check(candidate)")
    config = {
        "starter_code": native["starter_code"],
        "entry_point": native["entry_point"],
        "test_source": native["test"],
        "task_id": native["task_id"],
    }
    package = script_package(
        LCB_GRADE,
        config,
        environment=required_grader_environment(context),
        timeout=GRADER_TIMEOUT,
        answer_path=DEFAULT_OUTPUT,
    )
    events = (TextMessage(role="user", content=native["prompt"] + COMPLETION_INSTRUCTION),)
    task = conversation_task(row, events=events, package=package).model_copy(
        update={
            "tags": ("train", "code", "test-runner", "leetcode"),
            "labels": {"source_id": SOURCE_ID, "native_task_id": native["task_id"], "difficulty": native["difficulty"]},
        }
    )
    change = NormalizationChange(
        field="context.events",
        reason=(
            "Append the completion format instruction to the source prompt; the harness and starter code stay verbatim."
        ),
        original="The native prompt asks for a correct Python solution without fixing a reply format.",
        replacement=(
            "The task adds the fenced-block completion instruction so the grader can extract the candidate's code."
        ),
    )
    return NormalizedTask(task, (change,))


def source(files: tuple[str, ...]) -> RlDataSource:
    return RlDataSource(
        info=SourceInfo(
            origin=SourceOrigin.NEWFACADE,
            upstream_dataset="LeetCodeDataset",
            upstream_name="LeetCodeDataset",
            build_path=VERSION,
            family="code",
            data_restriction=DataRestriction.OPEN,
            task_format=TaskFormat.TASKSPEC,
            count=TRAIN_COUNT,
            notes=(
                "The source's own check(candidate) test harness is the grader: the context is the verbatim native "
                "prompt (problem statement and starter code), the candidate completes it in a fenced block, and the "
                "shipped runner execs the starter code and the candidate's block, binds the source entry point and runs "
                "the shipped harness verbatim. Reward 1 when every harness assertion passes, 0 on any candidate-code "
                "failure; harness defects exit nonzero as infrastructure errors. The reference completion stays "
                "private provenance, never judge gold. The train split (2,641 rows) is the population; the test "
                "split (228 rows) stays held out as shipped. The completion format instruction is recorded as a "
                "normalization. No live execution is claimed by conversion."
            ),
        ),
        config=CurationRecipe(
            name="leetcode-harness-grader",
            version=VERSION,
            source=HfSource(REPO, REVISION, files, SourceFormat.JSONL),
            grader=Environment(image=GRADER_IMAGE),
            convert=convert_problem,
            intended_use=IntendedUse.TRAIN,
        ),
        pipeline=process_rows,
    )
