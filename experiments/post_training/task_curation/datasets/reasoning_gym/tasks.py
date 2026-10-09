# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reasoning Gym puzzles from two sources, each graded by the puzzle task's own scorer.

``reasoning_gym_generated`` generates entries with the pinned ``reasoning_gym`` wheel, the release
the grader packages pin, in ``GENERATOR_PARTS`` parts that workers generate in parallel; a sample
generates only its sampled rows. Its grader
(``reasoning_gym_grade.py``) regenerates each entry before scoring, so the scorer sees the generator's
Python values, and rows whose entry a fresh dataset does not reproduce are rejected.
``tasktrove-reasoning-gym`` keeps the TaskTrove archive's ``tests/test.sh``, which thresholds the
scorer's reward at 0.5. Both run with the grader packages (``GRADER_PACKAGES``).
The Nemotron Ultra ``reasoning_gym`` component is declared with the other Ultra components.
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
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

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.tasktrove.archives import ANSWER_FILE_DELIVERY, tasktrove_source
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim, UrlSource
from experiments.post_training.task_curation.source import DataSourceMetadata, RlDataSource

SKYRL_METADATA = DataSourceMetadata(id="", name="", origin="MarinSkyRL", recorded_at="2026-10-08")
TASKTROVE_METADATA = DataSourceMetadata(id="", name="", origin="Task Trove", recorded_at="2026-10-08")

HERE = Path(__file__).parent
GENERATE = HERE / "generate.py"
# The grader lock (``datasets/grader.lock``) pins this wheel by the same hash, so the grader regenerates
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
        "worker and the grader's environment",
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


@contextmanager
def _generator_environment(wheel_path: StoragePath, python_hash_seed: int) -> Iterator[dict[str, str]]:
    """Unpack the generator wheel and yield the environment that puts it first on ``PYTHONPATH``."""
    with TemporaryDirectory() as directory:
        local_wheel = os.path.join(directory, GENERATOR_ARCHIVE)
        with wheel_path.open("rb") as source, open(local_wheel, "wb") as destination:
            shutil.copyfileobj(source, destination)
        packages = os.path.join(directory, "packages")
        with zipfile.ZipFile(local_wheel) as wheel:
            wheel.extractall(packages)
        yield {
            **os.environ,
            "PYTHONPATH": packages + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "MPLCONFIGDIR": directory,
            "PYTHONHASHSEED": str(python_hash_seed),
        }


def _generator_lines(arguments: list[str], environment: dict[str, str]) -> Iterator[str]:
    """Yield the output lines of ``generate.py``, raising with the end of its stderr when it fails."""
    with TemporaryFile(mode="w+b") as errors:
        process = subprocess.Popen(
            [sys.executable, str(GENERATE), *arguments],
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            env=environment,
        )
        try:
            assert process.stdout is not None
            yield from process.stdout
            if process.wait() != 0:
                errors.seek(0, os.SEEK_END)
                errors.seek(max(0, errors.tell() - GENERATOR_ERROR_BYTES))
                detail = errors.read().decode(errors="replace")
                raise RuntimeError(f"Pinned reasoning-gym generator failed: {detail}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def generated_row_count(
    wheel_path: StoragePath, excluded_generators: tuple[tuple[str, str], ...], python_hash_seed: int
) -> int:
    """The number of rows a whole run of the generator yields, from its pinned task registry."""
    with _generator_environment(wheel_path, python_hash_seed) as environment:
        (line,) = _generator_lines(["size", json.dumps(dict(excluded_generators))], environment)
    return int(line)


def generated_rows(
    wheel_path: StoragePath,
    generator_version: str,
    excluded_generators: tuple[tuple[str, str], ...],
    python_hash_seed: int,
    part: int,
    parts: int,
    indices: frozenset[int] | None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Run one part of ``generate.py`` with the unpacked generator wheel first on ``PYTHONPATH``.

    Yields the part's rows with their indices in the whole generated source, only those in
    ``indices`` when given.
    """
    arguments = ["rows", generator_version, json.dumps(dict(excluded_generators)), str(part), str(parts)]
    if indices is not None:
        arguments += ["--indices", json.dumps(sorted(indices))]
    with _generator_environment(wheel_path, python_hash_seed) as environment:
        for line in _generator_lines(arguments, environment):
            record = json.loads(line)
            yield record["index"], record["row"]


@dataclass(frozen=True)
class GeneratedRows:
    """Read the generator wheel by running the pinned generator in ``count`` parts."""

    generator_version: str
    excluded_generators: tuple[tuple[str, str], ...]
    python_hash_seed: int
    count: int

    def size(self, wheel_path: StoragePath, _context: ConversionContext) -> int:
        return generated_row_count(wheel_path, self.excluded_generators, self.python_hash_seed)

    def __call__(
        self, wheel_path: StoragePath, _context: ConversionContext, part: int, indices: frozenset[int] | None
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        return generated_rows(
            wheel_path,
            self.generator_version,
            self.excluded_generators,
            self.python_hash_seed,
            part,
            self.count,
            indices,
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
    """A reply task graded by the archive's ``tests/test.sh`` with the grader packages' reasoning-gym release."""
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


def sources() -> list[RlDataSource]:
    return [
        RlDataSource(
            metadata=replace(
                SKYRL_METADATA,
                id="MarinSkyRL:reasoning_gym",
                name="reasoning_gym",
                display_name="open-thought/reasoning-gym",
                url="https://github.com/open-thought/reasoning-gym",
                dataset_id="open-thought/reasoning-gym",
                revision="e44c4bfcb62c489286a1264094e6d9c883aaf0d2",
                revised_at="2026-10-08T02:13:45Z",
                dataset_revision="49b07130b3fcd12f2d064bba7c43869543a0e7e7",
                verifier_revision="c4daad3876de66d27fa1a5c6405269165f4ccfc5bd85e4fc65108a802f8d6109",
                family="reasoning-gym",
                environment="reasoning_gym",
                type="RLVR",
                turns="Single-turn",
                task_count=None,
                count_basis="Generated on demand; depends on selected tasks and rows_per_task",
                count_precision="not-applicable",
                count_url=(
                    "https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43"
                    "869543a0e7e7/README.md"
                ),
                kind="Generator",
                split="generated",
                benchmark_basis=(
                    "SkyRL test-only designation or HF benchmark:official tag; false means no " "designation found"
                ),
                family_basis="Upstream card/schema and selected SkyRL loader audited 2026-09-28",
                family_url=(
                    "https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43"
                    "869543a0e7e7/README.md"
                ),
                classification_basis=(
                    "Inferred from SkyRL environment contract; blended sources may contain " "multiple task types"
                ),
                canonical_source="open-thought/reasoning-gym",
                canonical_url="https://github.com/open-thought/reasoning-gym",
                provenance_url=(
                    "https://github.com/marin-community/MarinSkyRL/blob/e44c4bfcb62c489286a1264094"
                    "e6d9c883aaf0d2/infra/rl_data/sources.py"
                ),
                verification="two_sided",
                snapshot_safe=True,
                gym_alias="gym/reasoning_gym",
                gym_url=(
                    "https://github.com/marin-community/MarinSkyRL/blob/e44c4bfcb62c489286a1264094e6d"
                    "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/__init__.py"
                ),
                gym_entrypoint="skyrl_gym.envs.reasoning_gym.env:ReasoningGymEnv",
                dataset_revised_at="2026-04-17T19:39:15Z",
                registry_revised_at="2026-10-01T14:18:17Z",
                verifier_url=(
                    "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e"
                    "6d9c883aaf0d2/skyrl-gym/skyrl_gym/envs/reasoning_gym"
                ),
                verifier_revised_at="2026-10-08T02:13:45Z",
                revision_basis="Latest upstream dataset repository or MarinSkyRL verifier change",
            ),
            pipeline=RlDataPipeline(
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
                grader=GRADER_PACKAGES,
            ),
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-reasoning-gym-v2",
                name="laion__nemotron-gym-reasoning-gym-v2",
                display_name="laion/nemotron-gym-reasoning-gym-v2",
                url="https://huggingface.co/datasets/open-athena/task-trove",
                dataset_id="open-athena/task-trove",
                revision="ec049a4fb541ffbe5bbccb803e826563f5718dbf",
                revised_at="2026-10-08T09:34:47.000Z",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="reasoning-gym",
                environment="Harbor",
                type="Agentic",
                turns="Multi-turn",
                task_count=13712,
                count_basis="Released Harbor tasks: manifest by_source.converted",
                count_precision="exact",
                count_url=(
                    "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bb"
                    "ccb803e826563f5718dbf/manifest.json"
                ),
                notes="reasoning_gym library scoring is sound; remove the substring fallback at conversion.",
                benchmark_basis="Release manifest does not designate benchmarks",
                family_basis="Task Trove release manifest source_verdicts.family",
                family_url=(
                    "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bb"
                    "ccb803e826563f5718dbf/manifest.json"
                ),
                classification_basis="Task Trove tasks run as Agentic interactions in Harbor",
                canonical_source="laion/nemotron-gym-reasoning-gym-v2",
                canonical_url="https://huggingface.co/datasets/open-athena/task-trove",
                provenance_url=(
                    "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5"
                    "bbccb803e826563f5718dbf/manifest.json"
                ),
                verification="reasoning-gym",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-reasoning-gym-v2",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-reasoning-gym-v2",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=14259,
                modes=("reasoning-gym",),
            ),
            pipeline=RlDataPipeline(
                name="tasktrove-reasoning-gym",
                source=tasktrove_source(TASKTROVE_CONFIG),
                convert=convert_tasktrove,
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.TRAIN,
                rubric=TASKTROVE_RUBRIC,
                controls=TASKTROVE_CONTROLS,
                grader=GRADER_PACKAGES,
            ),
        ),
    ]
