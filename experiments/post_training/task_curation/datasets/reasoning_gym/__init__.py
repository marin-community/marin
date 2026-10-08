# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reasoning Gym puzzles from three sources, each graded by the puzzle task's own scorer.

``reasoning_gym_generated`` regenerates entries from a pinned reasoning-gym checkout; its grader
regenerates each entry again before scoring, so the scorer sees the generator's Python values.
``tasktrove-reasoning-gym`` keeps the TaskTrove archive's ``tests/test.sh``, which thresholds the
scorer's reward at 0.5. The Nemotron Ultra ``reasoning_gym`` components are scored on the reply's
last ``<answer>`` block or boxed answer. Every grader runs in ``REASONING_GYM_IMAGE``, which carries
one reasoning-gym package per source.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import Any

from pydantic import BaseModel, ValidationError
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.delivery import replace_phrases, rewritten_task
from taskcompendium.convert.nemotron_ultra import blend_task, text_request
from taskcompendium.convert.source_scorer import ANSWER_PATH, grade_script_package
from taskcompendium.convert.tasktrove import archive_resources, archive_script_grader
from taskcompendium.grader import GraderPackage, grader_config
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.inputs import SourceFormat, StagedInputs
from taskcompendium.pipeline.models import (
    Controls,
    ImportFailureKind,
    ImportRejection,
    IntendedUse,
    NormalizedTask,
    RawRow,
    Reply,
)
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.tasktrove import ANSWER_FILE_DELIVERY, tasktrove_source
from experiments.post_training.task_curation.images import REASONING_GYM_IMAGE
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim, UrlSource

GENERATOR_REVISION = "49b07130b3fcd12f2d064bba7c43869543a0e7e7"
GENERATOR_URL = f"https://api.github.com/repos/open-thought/reasoning-gym/tarball/{GENERATOR_REVISION}"
GENERATOR_SHA256 = "015beea52e7a1fa2f044827e054bcc36149abe8f55ebae233163735457dfbe98"
GENERATOR_ARCHIVE = "generator.tar.gz"
# generate.py runs as a script, so its bytes are not part of the declaration identity; bump the
# generated pipeline's version when it changes.
GENERATE = Path(__file__).with_name("generate.py")
GENERATOR_ERROR_BYTES = 8192
EXCLUDED_GENERATORS = (("composite", "Orchestration constructor requires explicit component DatasetSpec configuration"),)
PYTHON_HASH_SEED = 0

TASKTROVE_CONFIG = "laion__nemotron-gym-reasoning-gym-v2"
ULTRA_AGENT = "reasoning_gym_simple_agent"
# Each source's reasoning-gym release is installed separately in the grading image.
TASKTROVE_PACKAGE = "/opt/reasoning-gym-tasktrove"
ULTRA_PACKAGE = "/opt/reasoning-gym-ultra"
GENERATED_PACKAGE = "/opt/reasoning-gym-generated"
SKYRL_GYM_PACKAGE = "/opt/skyrl_gym"
GRADE = "reasoning_gym_grade.py"
GRADE_BYTES = Path(__file__).with_name(GRADE).read_bytes()
SCORER_TIMEOUT = 60.0
ANSWER_FILE_NOTE = f"\nThe runtime writes your final assistant response to {ANSWER_PATH}."
REWRITE_REASON = f"The grader reads the reply, which the runtime writes to {ANSWER_PATH}"

GENERATED_RUBRIC = """
Read the complete generated question and verify that every grid, rule, sequence, or example required to solve it is
present.

Independently check the hidden generated answer against the public problem where feasible; a passing score does not
establish correctness.

Preserve the task's own partial reward and answer parsing; a label or approximate substring match is not a substitute
for that scorer.

Judge default generated difficulty and unusual puzzle formats on their actual content, without assuming they are
defects.

Check non-discrimination when a scorer accepts incorrect alternatives, and record concrete semantic mismatches.
"""

TASKTROVE_RUBRIC = """
The whole problem must be comprehensible and provide every grid, sequence, statement, or rule needed to answer it.
Procedural generation and unfamiliar puzzles are not defects by themselves.

Independently work out the answer where feasible; check that the reference actually follows the public problem.
Passing a reference through its scorer tests mechanics, not its truth.

Compare the public answer format with the named upstream Reasoning Gym scorer. The source grader thresholds the
scorer's reward at 0.5 into a binary verdict and keeps the scorer's answer parsing. The source fallback is reachable
only when its source guard and validation permit it.

Flag multiple defensible answers when the named scorer rejects them, hidden assumptions, underspecified
transformations, or a hidden question different from the public problem.
"""


def generated_rows(
    archive_path: StoragePath,
    generator_revision: str,
    excluded_generators: tuple[tuple[str, str], ...],
    python_hash_seed: int,
) -> Iterator[dict[str, Any]]:
    """Run ``generate.py`` against the unpacked generator archive and yield its JSONL rows."""
    with TemporaryDirectory() as directory:
        local_archive = os.path.join(directory, GENERATOR_ARCHIVE)
        with archive_path.open("rb") as source, open(local_archive, "wb") as destination:
            shutil.copyfileobj(source, destination)
        with tarfile.open(local_archive, mode="r:gz") as archive:
            roots = {member.name.split("/", 1)[0] for member in archive if member.name}
            if len(roots) != 1:
                raise ValueError("Pinned generator archive has multiple roots")
            archive.extractall(directory, filter="data")
        root = os.path.join(directory, roots.pop())
        environment = {
            **os.environ,
            "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "MPLCONFIGDIR": directory,
            "PYTHONHASHSEED": str(python_hash_seed),
        }
        with TemporaryFile(mode="w+b") as errors:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(GENERATE),
                    generator_revision,
                    json.dumps(dict(excluded_generators)),
                    str(python_hash_seed),
                ],
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                env=environment,
            )
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    yield json.loads(line)
                if process.wait() != 0:
                    errors.seek(0, os.SEEK_END)
                    errors.seek(max(0, errors.tell() - GENERATOR_ERROR_BYTES))
                    detail = errors.read().decode(errors="replace")
                    raise RuntimeError(f"Pinned reasoning-gym generator failed: {detail}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()


@dataclass(frozen=True)
class GeneratedRows:
    """Read the generator archive by running the pinned generator."""

    generator_revision: str
    excluded_generators: tuple[tuple[str, str], ...]
    python_hash_seed: int

    def __call__(self, archive_path: StoragePath, _inputs: StagedInputs) -> Iterator[dict[str, Any]]:
        return generated_rows(archive_path, self.generator_revision, self.excluded_generators, self.python_hash_seed)


class RecordedScoringError(BaseModel):
    type: str
    message: str


class RecordedReward(BaseModel):
    candidate: str | None
    reward: float | None
    scoring_error: RecordedScoringError | None = None


class RecordedControls(BaseModel):
    """The scorer's rewards for the known answer and a fixed wrong answer, recorded at generation."""

    generator_revision: str
    positive: RecordedReward
    negative: RecordedReward
    execution: str


def _scorer_package(mode: str, contract: dict[str, Any], package_path: str) -> GraderPackage:
    return grade_script_package(
        GRADE,
        GRADE_BYTES,
        config={"mode": mode, "contract": contract},
        environment=REASONING_GYM_IMAGE.requirements(),
        timeout=SCORER_TIMEOUT,
        env={"PYTHONPATH": f"{package_path}:{SKYRL_GYM_PACKAGE}"},
    )


def convert_generated(row: RawRow) -> TaskSpec | ImportRejection:
    try:
        entry = row.data["entry"]
        generation = row.data["generation"]
        task_name = entry["metadata"]["source_dataset"]
        if task_name != generation["task"] or (entry["answer"] is not None and not isinstance(entry["answer"], str)):
            raise ValueError("Generated entry and generation provenance must identify the same task")
        recorded = RecordedControls.model_validate(row.data["recorded_pinned_generator_controls"])
    except (ValidationError, ValueError, KeyError, TypeError) as error:
        return ImportRejection(
            kind=ImportFailureKind.CONVERTER_ERROR, reason="invalid_generated_reasoning_entry", detail=str(error)
        )
    contract = {
        "task": task_name,
        "entry": entry,
        "generation": generation,
        "generator_revision": GENERATOR_REVISION,
        "answer_extraction": "Text after the last Answer: marker, stripped; otherwise stripped whole response",
        "reward": "float(reasoning_gym.get_score_answer_fn(task)(answer, entry))",
        "recorded_pinned_generator_controls": recorded.model_dump(mode="json"),
    }
    package = _scorer_package("generated", contract, GENERATED_PACKAGE)
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=entry["question"]),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=package.grader,
        resources=ResourceGroups(verifier=package.resources),
    )


def generated_golden(task: TaskSpec) -> Reply | None:
    """The generator's recorded known answer, when the task has one."""
    candidate = grader_config(task)["contract"]["recorded_pinned_generator_controls"]["positive"]["candidate"]
    return answer_reply(task, candidate) if candidate is not None else None


def reply_instruction(instruction: str) -> str:
    """Ask for the answer in the reply where the instruction asked for ``/app/answer.txt``."""
    public = replace_phrases(instruction, ANSWER_FILE_DELIVERY)
    if public == instruction and ANSWER_PATH in instruction:
        # Unrecognized wording: keep it, and say where the reply ends up.
        public += ANSWER_FILE_NOTE
    return public


def convert_tasktrove(row: RawRow) -> TaskSpec | NormalizedTask | ImportRejection:
    """A reply task graded by the archive's ``tests/test.sh`` with the TaskTrove reasoning-gym release."""
    instruction, data = row.data.get("instruction"), row.data.get("verifier_data")
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(data, dict):
        return source_defect("missing_input", "Instruction and entry data are required")
    metadata = data.get("metadata")
    dataset = metadata.get("source_dataset") if isinstance(metadata, dict) else None
    if not isinstance(dataset, str) or not dataset:
        return unsupported("missing_scorer", "metadata.source_dataset is required")
    if not isinstance(data.get("answer"), str):
        return source_defect("invalid_entry", "Entry answer must be a string")
    grader = archive_script_grader(
        row.data,
        required=("tests/verifier.py",),
        environment=REASONING_GYM_IMAGE.requirements(),
        answer_path=ANSWER_PATH,
        env={"PYTHONPATH": TASKTROVE_PACKAGE},
    )
    if isinstance(grader, ImportRejection):
        return grader
    task = TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=reply_instruction(instruction)),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader,
        resources=archive_resources(row.data),
    )
    return rewritten_task(task, original=instruction, reason=REWRITE_REASON)


def tasktrove_golden(task: TaskSpec) -> Reply:
    entry = json.loads(
        resource_bytes(next(item for item in task.resources.verifier if item.path == "verifier_data.json"))
    )
    return answer_reply(task, entry["answer"])


def convert_ultra_reasoning_gym(row: RawRow) -> NormalizedTask | ImportRejection:
    request = text_request(row.data, (ULTRA_AGENT,))
    if isinstance(request, ImportRejection):
        return request
    return blend_task(row, request, _scorer_package("ultra", request.contract, ULTRA_PACKAGE))


def ultra_golden(task: TaskSpec) -> Reply | None:
    answer = grader_config(task)["contract"].get("answer")
    return answer_reply(task, answer) if isinstance(answer, str) else None


# Scorers give partial credit, so a fixed wrong answer has no single expected reward; only the
# known answer is checked.
GENERATED_CONTROLS = Controls(golden=generated_golden)
TASKTROVE_CONTROLS = Controls(golden=tasktrove_golden)
ULTRA_CONTROLS = Controls(golden=ultra_golden)


def pipelines() -> list[RlDataPipeline]:
    return [
        RlDataPipeline(
            name="reasoning_gym_generated",
            source=UrlSource(
                GENERATOR_URL,
                GENERATOR_SHA256,
                GENERATOR_ARCHIVE,
                SourceFormat.GENERATED,
                read=GeneratedRows(GENERATOR_REVISION, EXCLUDED_GENERATORS, PYTHON_HASH_SEED),
            ),
            convert=convert_generated,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=GENERATED_RUBRIC,
            controls=GENERATED_CONTROLS,
            atlas_id="MarinSkyRL:reasoning_gym",
        ),
        RlDataPipeline(
            name="tasktrove-reasoning-gym",
            source=tasktrove_source(TASKTROVE_CONFIG),
            convert=convert_tasktrove,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=TASKTROVE_RUBRIC,
            controls=TASKTROVE_CONTROLS,
            atlas_id=f"Task Trove:{TASKTROVE_CONFIG}",
        ),
    ]
