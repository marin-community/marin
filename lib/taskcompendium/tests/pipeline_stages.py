# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixture sources, converters and grading machines that exercise the curation stages."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, DockerImage, Machine, MachineFactory, MachineSpec, ShellSimBuiltins

from taskcompendium.convert.answers import numeric_answer_task, source_defect
from taskcompendium.models import (
    EnvironmentRequirements,
    ExitCodeReward,
    ScriptGrader,
    Source,
    TaskSpec,
)
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import (
    Controls,
    Converter,
    FilterPolicy,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    SourceRecipe,
)
from taskcompendium.pipeline.review import BatchReviewer
from taskcompendium.pipeline.stages import (
    AuditExecution,
    ReviewConfig,
    ReviewMode,
    audit_prepared_source,
    filter_source,
    prepare_source,
)
from taskcompendium.runtime.resources import inline_resource

SOURCE_FILES = SourceFiles("fixture/tasks", "1", ("source.jsonl",), SourceFormat.JSONL)
SVAMP_RUBRIC = ReviewRubric(
    id="arithmetic-word-problems",
    version="1",
    criteria=(
        "Identify the quantities and the operation the question actually requests. Check units and directionality.",
        "Flag contradictions or missing quantities that prevent a unique numeric answer.",
    ),
)
GRADER_IMAGE = "fixture@sha256:" + "b" * 64
GRADER_ENVIRONMENT = EnvironmentRequirements(docker_image=GRADER_IMAGE, compatible_backends=(Backend.DOCKER,))


def svamp_row_task(row: RawRow) -> TaskSpec | ImportRejection:
    """An arithmetic word problem graded by its exact numeric answer; the equation stays private."""
    body, question = row.data.get("Body"), row.data.get("Question")
    if not isinstance(body, str) or not isinstance(question, str):
        return source_defect("missing_prompt", "Body and Question must be strings")
    return numeric_answer_task(
        row, prompt=f"{body.strip()} {question.strip()}", answer=row.data.get("Answer"), tolerance_abs=0, tolerance_rel=0
    )


def convert_svamp(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    return svamp_row_task(row)


def svamp_task(task_id: str, data: Mapping[str, Any]) -> TaskSpec:
    task = svamp_row_task(RawRow(task_id, Source(dataset="fixture", revision="1", row="0", importer_revision="1"), data))
    assert isinstance(task, TaskSpec)
    return task


def script_graded(task: TaskSpec, script: bytes, *, answer_path: str | None = "/app/answer.txt") -> TaskSpec:
    """``task`` graded in the fixture grader image by ``script``, which passes by exiting zero in /app."""
    grader = ScriptGrader(
        argv=("bash", "/tests/grade.sh"),
        environment=GRADER_ENVIRONMENT,
        answer_path=answer_path,
        reward=ExitCodeReward(),
        timeout=30,
    )
    resources = task.resources.model_copy(update={"verifier": (inline_resource("grade.sh", script),)})
    return TaskSpec.model_validate_json(
        task.model_copy(update={"grader": grader, "resources": resources}).model_dump_json()
    )


def fixture_recipe(
    convert: Converter,
    *,
    rubric: ReviewRubric | None = SVAMP_RUBRIC,
    controls: Controls | None = None,
    source: SourceFiles = SOURCE_FILES,
) -> SourceRecipe:
    return SourceRecipe(
        name="fixture",
        version="1",
        source=source,
        convert=convert,
        rubric=rubric,
        controls=controls,
        intended_use=IntendedUse.TRAIN,
    )


def review_config(reviewer: BatchReviewer) -> ReviewConfig:
    return ReviewConfig(
        reviewer.model,
        reviewer.model_revision,
        reviewer.max_prompt_characters,
        reviewer.max_tokens,
        reviewer.max_attempts,
        reviewer.retry_max_tokens,
        reviewer.retry_max_prompt_characters,
        max_batch_bytes=reviewer.max_batch_bytes,
        mode=ReviewMode.BATCH,
    )


def review_source(
    staged: str, output: str, recipe: SourceRecipe, execution: AuditExecution, limit: int | None
) -> dict[str, Any]:
    """Prepare and review every eligible row, without source-level quality extrapolation."""
    assert isinstance(execution.reviewer, BatchReviewer)
    prepare_source(staged, output, recipe, limit, execution)
    return audit_prepared_source(output, None, output, recipe, review_config(execution.reviewer), execution).manifest


def run_stages(
    recipe: SourceRecipe,
    rows: Iterable[Mapping[str, Any]],
    *,
    output_path: Path,
    limit: int,
    reviewer: BatchReviewer,
    policy: FilterPolicy = FilterPolicy(),
) -> dict[str, Any]:
    staged, audited, filtered = (output_path / name for name in ("staged", "audited", "filtered"))
    source = staged / "source.jsonl"
    if not source.exists():
        staged.mkdir(parents=True)
        source.write_text("".join(json.dumps(dict(row)) + "\n" for row in rows))
    if not (audited / "manifest.json").exists():
        review_source(str(staged), str(audited), recipe, AuditExecution(reviewer=reviewer), limit)
    return filter_source(str(audited), str(filtered), policy, max_workers=1)


def stage_table(output_path: Path, view: str = "audit") -> pa.Table:
    files = sorted((output_path / "filtered" / view).glob("*.parquet"))
    table = pa.concat_tables([pq.read_table(file) for file in files])
    rows = sorted(
        table.to_pylist(),
        key=lambda row: (row["source_row"].rsplit(":", 1)[0], int(row["source_row"].rsplit(":", 1)[1])),
    )
    return pa.Table.from_pylist(rows, schema=table.schema)


class ShellSimImages:
    """Stand in for a grader image with a fresh in-memory ShellSim machine and its shell builtins."""

    backend = Backend.DOCKER

    async def create(self, spec: MachineSpec) -> Machine:
        return await ShellSimMachineFactory().create(replace(spec, source=ShellSimBuiltins()))


class UnavailableImages:
    backend = Backend.DOCKER

    async def create(self, spec: MachineSpec) -> Machine:
        raise RuntimeError("Machine service unavailable")


@dataclass
class FixtureGradingMachines:
    """Grading machines for a campaign whose machine service is ``factory``."""

    factory: MachineFactory = field(default_factory=ShellSimImages)

    def identity(self) -> dict[str, Any]:
        return {"backend": "fixture", "network": "deny"}

    def machine(self, image: str, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        return self.factory, MachineSpec(DockerImage(image), memory_mb=memory_mb)
