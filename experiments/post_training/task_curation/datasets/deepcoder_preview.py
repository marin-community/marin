# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert DeepCoder-Preview's stdin/stdout configs with the shipped APPS evaluator.

The preview dataset's primeintellect and taco train configs carry stdin/stdout test pairs
(the codeforces/lcbv5 splits stay out of this population: codeforces is the paper's eval
split and lcbv5 is mostly function-completion, a separate follow-up). The tests strings
normalize per config into the APPS ``input_output`` contract and the shipped APPS runner
scores a candidate's last fenced block: reward 1 when every case passes. The source's
reference solutions stay private provenance — never judge gold, never in the grader config.
"""

import json

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.script_grader import script_package
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.models import TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizationChange, NormalizedTask, RawRow
from verifyit.spec import DEFAULT_OUTPUT

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.skyrl.code import APPS_GRADE, CODE_INSTRUCTION
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

REPO = "agentica-org/DeepCoder-Preview-Dataset"
REVISION = "177913a7bd43791646ef6a43645caa3c871ab3db"
SOURCE_ID = "agentica-org:DeepCoder-Preview-Dataset"
VERSION = "deepcoder-apps-grader-v1"
COUNT = 23688
FILES = (
    *(f"primeintellect/train-0000{i}-of-00005.parquet" for i in range(5)),
    *(f"taco/train-0000{i}-of-00004.parquet" for i in range(4)),
)


def normalize_tests(native_tests: str) -> dict | ImportRejection:
    """Normalize the per-config tests string into the APPS ``{"inputs","outputs"}`` contract.

    codeforces/primeintellect/lcbv5 ship a JSON array of ``{"input","output"}`` cases;
    taco ships the paired dict directly. Anything else fails closed.
    """
    try:
        decoded = json.loads(native_tests)
    except json.JSONDecodeError as error:
        return source_defect("invalid_test_contract", f"tests is not JSON: {error}")
    if isinstance(decoded, dict):
        cases = decoded
    elif isinstance(decoded, list):
        if not decoded or not all(isinstance(case, dict) and "input" in case and "output" in case for case in decoded):
            return source_defect("invalid_test_contract", "the tests array must carry input/output cases")
        cases = {"inputs": [case["input"] for case in decoded], "outputs": [case["output"] for case in decoded]}
    else:
        return source_defect("invalid_test_contract", "tests must be an array of cases or a paired dict")
    inputs, outputs = cases.get("inputs"), cases.get("outputs")
    if not isinstance(inputs, list) or not inputs or not isinstance(outputs, list) or len(inputs) != len(outputs):
        return source_defect("invalid_test_contract", "Paired nonempty inputs and outputs are required")
    try:
        json.dumps(cases, allow_nan=False)
    except ValueError:
        return unsupported("nonfinite_test_contract", "tests contain a non-finite number")
    return cases


def convert_problem(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    native = dict(row.data)
    required = {"problem", "tests"}
    if not required <= set(native) or not isinstance(native["problem"], str) or not native["problem"].strip():
        return source_defect("incomplete_deepcoder_contract", "The task fields with a nonempty problem are required")
    if not isinstance(native["tests"], str) or not native["tests"].strip():
        return source_defect("incomplete_deepcoder_contract", "tests must be a nonempty JSON string")
    cases = normalize_tests(native["tests"])
    if isinstance(cases, ImportRejection):
        return cases
    package = script_package(
        APPS_GRADE,
        {"input_output": cases},
        environment=required_grader_environment(context),
        timeout=330.0,
        answer_path=DEFAULT_OUTPUT,
    )
    events = (TextMessage(role="user", content=native["problem"] + CODE_INSTRUCTION),)
    task = conversation_task(row, events=events, package=package).model_copy(
        update={
            "tags": ("train", "code", "stdin-stdout", "deepcoder"),
            "labels": {"source_id": SOURCE_ID},
        }
    )
    change = NormalizationChange(
        field="context.events",
        reason="Append the completion format instruction to the verbatim problem; the problem statement stays verbatim.",
        original="The native problem asks for a correct program without fixing a reply format.",
        replacement=(
            "The task adds the fenced-block completion instruction so the shipped APPS runner can extract "
            "the candidate's program."
        ),
    )
    return NormalizedTask(task, (change,))


def source(files: tuple[str, ...]) -> RlDataSource:
    return RlDataSource(
        info=SourceInfo(
            origin=SourceOrigin.AGENTICA_ORG,
            upstream_dataset="DeepCoder-Preview-Dataset",
            upstream_name="DeepCoder-Preview-Dataset",
            build_path=VERSION,
            family="code",
            data_restriction=DataRestriction.OPEN,
            task_format=TaskFormat.TASKSPEC,
            count=COUNT,
            notes=(
                "TRAIN code tasks from the DeepCoder-Preview preview dataset's stdin/stdout configs "
                "(primeintellect 16,252 + taco 7,436): the context is the verbatim problem plus a recorded "
                "completion-format normalization, and the grader is the shipped APPS runner over the "
                "normalized input/output cases (reward 1 when every case passes, 0 on candidate-code "
                "failure; harness defects exit nonzero as infrastructure errors). The source's reference "
                "solutions stay private provenance — never judge gold and never shipped in the grader "
                "config. The codeforces eval split and the lcbv5 config stay out of this population "
                "(the paper's eval basis and a mostly function-completion follow-up respectively). "
                "Contract defects fail closed. No live execution is claimed by conversion."
            ),
        ),
        config=CurationRecipe(
            name="deepcoder-apps-grader",
            version=VERSION,
            source=HfSource(REPO, REVISION, files, SourceFormat.PARQUET),
            grader=GRADER_PACKAGES,
            convert=convert_problem,
            intended_use=IntendedUse.TRAIN,
        ),
        pipeline=process_rows,
    )
