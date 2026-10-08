# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reasoning Gym puzzles from two sources, each graded by the puzzle task's own scorer.

``reasoning_gym_generated`` generates entries with the pinned ``reasoning_gym`` wheel, the release
the grader image installs, in ``GENERATOR_PARTS`` parts that workers generate in parallel. Its grader
(``reasoning_gym_grade.py``) regenerates each entry before scoring, so the scorer sees the generator's
Python values, and rows whose entry a fresh dataset does not reproduce are rejected.
``tasktrove-reasoning-gym`` keeps the TaskTrove archive's ``tests/test.sh``, which thresholds the
scorer's reward at 0.5. Both run in the grader image (``images.recipes.GRADER``).
The Nemotron Ultra ``reasoning_gym`` component is declared with the other Ultra components.
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import Any

from pydantic import BaseModel, ValidationError
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.delivery import replace_phrases, rewritten_task
from taskcompendium.convert.script_grader import grade_script, script_package, shipped_files
from taskcompendium.convert.tasktrove import ANSWER_PATH, archive_resources, archive_script_grader
from taskcompendium.grader import grader_config
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
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
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

from experiments.post_training.task_curation.datasets.tasktrove.archives import ANSWER_FILE_DELIVERY, tasktrove_source
from experiments.post_training.task_curation.images.recipes import GRADER
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim, UrlSource

HERE = Path(__file__).parent
GENERATE = HERE / "generate.py"
# The grader image's requirements.lock pins this wheel by the same hash, so the grader regenerates
# entries with the code that generated them.
GENERATOR_VERSION = "0.1.25"
GENERATOR_ARCHIVE = f"reasoning_gym-{GENERATOR_VERSION}-py3-none-any.whl"
GENERATOR_URL = (
    "https://files.pythonhosted.org/packages/42/21/"
    f"bb9f6d2f76424f8b516fe83f8026e52e466b0202341bb4ee7a9861e06277/{GENERATOR_ARCHIVE}"
)
GENERATOR_SHA256 = "7f17a3eddb13c015d7d4a755ed576a061df889faf9468bcc2cca334ebe9e0435"
GENERATOR_ERROR_BYTES = 8192
EXCLUDED_GENERATORS = (
    ("composite", "Orchestration constructor requires explicit component DatasetSpec configuration"),
    (
        "figlet_font",
        "Picks fonts by position in pyfiglet's font directory listing, whose order differs between the generating "
        "worker and the grader image",
    ),
)
PYTHON_HASH_SEED = 0
GENERATOR_PARTS = 32
"""Parts of the task registry generated on separate workers; the slowest task bounds each part."""
GENERATED_GRADE = grade_script(HERE / "reasoning_gym_grade.py", *shipped_files(HERE, GENERATE.name))

TASKTROVE_CONFIG = "laion__nemotron-gym-reasoning-gym-v2"
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
    wheel_path: StoragePath,
    generator_version: str,
    excluded_generators: tuple[tuple[str, str], ...],
    python_hash_seed: int,
    part: int,
    parts: int,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Run one part of ``generate.py`` with the unpacked generator wheel first on ``PYTHONPATH``.

    Yields the part's rows with their indices in the whole generated source.
    """
    with TemporaryDirectory() as directory:
        local_wheel = os.path.join(directory, GENERATOR_ARCHIVE)
        with wheel_path.open("rb") as source, open(local_wheel, "wb") as destination:
            shutil.copyfileobj(source, destination)
        packages = os.path.join(directory, "packages")
        with zipfile.ZipFile(local_wheel) as wheel:
            wheel.extractall(packages)
        environment = {
            **os.environ,
            "PYTHONPATH": packages + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "MPLCONFIGDIR": directory,
            "PYTHONHASHSEED": str(python_hash_seed),
        }
        with TemporaryFile(mode="w+b") as errors:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(GENERATE),
                    generator_version,
                    json.dumps(dict(excluded_generators)),
                    str(part),
                    str(parts),
                ],
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                env=environment,
            )
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    record = json.loads(line)
                    yield record["index"], record["row"]
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
    """Read the generator wheel by running the pinned generator in ``count`` parts."""

    generator_version: str
    excluded_generators: tuple[tuple[str, str], ...]
    python_hash_seed: int
    count: int

    def __call__(
        self, wheel_path: StoragePath, _context: ConversionContext, part: int
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        return generated_rows(
            wheel_path, self.generator_version, self.excluded_generators, self.python_hash_seed, part, self.count
        )


class RecordedScoringError(BaseModel):
    type: str
    message: str


class RecordedReward(BaseModel):
    candidate: str | None
    reward: float | None
    scoring_error: RecordedScoringError | None = None


class RecordedControls(BaseModel):
    """The scorer's rewards for the known answer and a fixed wrong answer, recorded at generation."""

    generator_version: str
    positive: RecordedReward
    negative: RecordedReward
    execution: str


def convert_generated(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    try:
        entry = row.data["entry"]
        generation = row.data["generation"]
        reproducible = row.data["reproducible"]
        task_name = entry["metadata"]["source_dataset"]
        if task_name != generation["task"] or (entry["answer"] is not None and not isinstance(entry["answer"], str)):
            raise ValueError("Generated entry and generation provenance must identify the same task")
        recorded = RecordedControls.model_validate(row.data["recorded_pinned_generator_controls"])
    except (ValidationError, ValueError, KeyError, TypeError) as error:
        return ImportRejection(
            kind=ImportFailureKind.CONVERTER_ERROR, reason="invalid_generated_reasoning_entry", detail=str(error)
        )
    if reproducible is not True:
        return source_defect("irreproducible_entry", f"A fresh {task_name} dataset generates a different entry")
    contract = {
        "task": task_name,
        "entry": entry,
        "generation": generation,
        "generator_version": GENERATOR_VERSION,
        "answer_extraction": "Text after the last Answer: marker, stripped; otherwise stripped whole response",
        "reward": "float(reasoning_gym.get_score_answer_fn(task)(answer, entry))",
        "recorded_pinned_generator_controls": recorded.model_dump(mode="json"),
    }
    package = script_package(
        GENERATED_GRADE,
        {"contract": contract},
        environment=required_grader_environment(context),
        timeout=SCORER_TIMEOUT,
        answer_path=ANSWER_PATH,
        env={"PYTHONHASHSEED": str(generation["python_hash_seed"])},
    )
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


def convert_tasktrove(row: RawRow, context: ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection:
    """A reply task graded by the archive's ``tests/test.sh`` with the grader image's reasoning-gym release."""
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
        environment=required_grader_environment(context),
        answer_path=ANSWER_PATH,
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


GENERATED_CONTROLS = Controls(golden=generated_golden)
TASKTROVE_CONTROLS = Controls(golden=tasktrove_golden)


def pipelines() -> list[RlDataPipeline]:
    return [
        RlDataPipeline(
            name="reasoning_gym_generated",
            source=UrlSource(
                GENERATOR_URL,
                GENERATOR_SHA256,
                GENERATOR_ARCHIVE,
                SourceFormat.GENERATED,
                parts=GeneratedRows(GENERATOR_VERSION, EXCLUDED_GENERATORS, PYTHON_HASH_SEED, GENERATOR_PARTS),
            ),
            convert=convert_generated,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=GENERATED_RUBRIC,
            controls=GENERATED_CONTROLS,
            atlas_id="MarinSkyRL:reasoning_gym",
            grader_image=GRADER,
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
            grader_image=GRADER,
        ),
    ]
