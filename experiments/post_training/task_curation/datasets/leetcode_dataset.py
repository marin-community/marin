# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert LeetCodeDataset's train problems with the source's own test harness as the grader."""

import json
import subprocess
import sys
import tempfile
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
VERSION = "leetcode-harness-grader-v2"
TRAIN_COUNT = 2641
GRADER_IMAGE = (
    "ghcr.io/marin-community/iris-task@sha256:c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab"
)
GRADER_TIMEOUT = 330.0
QUALIFICATION_TIMEOUT = 120.0
COMPLETION_INSTRUCTION = "\nReturn the complete Python solution in this format:\n```python\n# solution\n```"
LCB_GRADE = grade_script(HERE / "leetcode_grade.py")

REFERENCE_QUALIFICATION_EVIDENCE: dict[str, str] = {}
"""Per-task evidence for the typed reference-qualification exclusions, sealed by the census runner."""

QUALIFICATION_DRIVER = """
import json, sys
sys.path.insert(0, sys.argv[1])
from leetcode_harness import HarnessDefect, harness_reward
try:
    reward = harness_reward(json.load(open(sys.argv[2])), open(sys.argv[3]).read())
except HarnessDefect as error:
    print(f"harness defect: {error}", file=sys.stderr)
    raise SystemExit(3)
except Exception as error:
    print(f"qualification driver defect: {error}", file=sys.stderr)
    raise SystemExit(4)
print(reward)
"""


def reference_qualification(native: dict) -> ImportRejection | None:
    """CPU check of the source's own reference solution against the shipped harness.

    A task whose harness cannot execute the source's own correct solution — a broken
    ``check()`` harness or a reference that fails its own tests — is a per-task source
    defect: typed rejection with evidence, never a grader edit (coordinator ruling
    2026-11-27). The evidence dict is sealed by the census runner.
    """
    config = {
        "preamble": native["prompt"],
        "starter_code": native["starter_code"],
        "entry_point": native["entry_point"],
        "test_source": native["test"],
        "task_id": native["task_id"],
    }
    with tempfile.TemporaryDirectory(prefix="leetcode-reference-qualification-") as temporary:
        root = Path(temporary)
        (root / "config.json").write_text(json.dumps(config))
        (root / "answer.txt").write_text(native["completion"])
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    QUALIFICATION_DRIVER,
                    str(HERE),
                    str(root / "config.json"),
                    str(root / "answer.txt"),
                ],
                capture_output=True,
                text=True,
                timeout=QUALIFICATION_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            REFERENCE_QUALIFICATION_EVIDENCE[native["task_id"]] = (
                f"the shipped harness timed out after {QUALIFICATION_TIMEOUT:.0f}s on the source reference"
            )
            return source_defect(
                "harness_defect",
                "The shipped harness times out on the source's own reference solution; see the sealed "
                "reference-qualification evidence",
            )
        if completed.returncode == 3 or completed.returncode == 4:
            REFERENCE_QUALIFICATION_EVIDENCE[native["task_id"]] = completed.stderr.strip()[-400:]
            return source_defect(
                "harness_defect",
                "The shipped harness cannot execute the source's own reference solution; see the sealed "
                "reference-qualification evidence",
            )
        if completed.returncode != 0 or completed.stdout.strip() != "1":
            REFERENCE_QUALIFICATION_EVIDENCE[native["task_id"]] = completed.stderr.strip()[-400:] or "reward 0"
            return source_defect(
                "harness_rejects_reference_solution",
                "The source's own reference solution fails the shipped check() harness; see the sealed "
                "reference-qualification evidence",
            )
    return None


def convert_problem(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    native = dict(row.data)
    required = {
        "prompt",
        "starter_code",
        "entry_point",
        "test",
        "task_id",
        "difficulty",
        "query",
        "response",
        "completion",
    }
    if not required <= set(native) or not isinstance(native["query"], str) or not native["query"].strip():
        return source_defect("incomplete_leetcode_contract", "The task fields with a nonempty query are required")
    for field in ("prompt", "starter_code", "entry_point", "test", "completion"):
        if not isinstance(native[field], str) or not native[field].strip():
            return source_defect("incomplete_leetcode_contract", f"{field} must be nonempty text")
    callable_name = native["entry_point"].replace("(", " ").split()[-1].split(".")[-1]
    if callable_name not in native["starter_code"]:
        return source_defect("entry_point_not_in_starter", "The entry point must name the starter code's callable")
    if "def check(" not in native["test"]:
        return source_defect("missing_check_harness", "The source test harness must define check(candidate)")
    rejection = reference_qualification(native)
    if rejection is not None:
        return rejection
    config = {
        "preamble": native["prompt"],
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
    events = (TextMessage(role="user", content=native["query"] + COMPLETION_INSTRUCTION),)
    task = conversation_task(row, events=events, package=package).model_copy(
        update={
            "tags": ("train", "code", "test-runner", "leetcode"),
            "labels": {"source_id": SOURCE_ID, "native_task_id": native["task_id"], "difficulty": native["difficulty"]},
        }
    )
    change = NormalizationChange(
        field="context.events",
        reason=(
            "Append the completion format instruction to the source's complete query framing; the harness and "
            "starter code stay verbatim."
        ),
        original=(
            "The native query is the source's own complete framing (expert-programmer wrapper, problem "
            "statement, format instruction and starter code) but does not fix a reply envelope."
        ),
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
                "query — the source's own complete framing, measured to carry the problem statement and starter "
                "code verbatim in every row (the v1 binding of the boilerplate prompt field was rejected by "
                "SAMPLE and is superseded) — the candidate completes it in a fenced block, and the "
                "shipped runner execs the source's own runtime preamble (the prompt's imports and helper "
                "classes, compiled in every native row; the truncated starter_code stays provenance and is "
                "never executed), the candidate's block, and the shipped harness verbatim. Every task passes a "
                "conversion-time CPU qualification: the source's own reference solution must pass the shipped "
                "harness; failures are typed per-task source-defect exclusions with sealed evidence, never "
                "grader edits. Reward 1 when every harness assertion passes, 0 on any candidate-code "
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
