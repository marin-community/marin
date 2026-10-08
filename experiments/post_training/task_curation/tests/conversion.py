# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert fixture rows through a declaration the way the source pipeline does."""

import io
import tarfile
from collections.abc import Mapping
from typing import Any

from taskcompendium.convert.environment import grading_environment
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext, StagedInputs
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask, RawRow
from taskcompendium.pipeline.transforms import row_source, row_task_id

from experiments.post_training.task_curation.pipeline import RlDataPipeline, source_recipe

FIXTURE_LOCATOR = "fixture.jsonl:0"
FIXTURE_GRADER_IMAGE = "ghcr.io/marin-community/task-curation-grader@sha256:" + "0" * 64
"""Stands in for the built grader image: conversion records the reference without running it."""
FIXTURE_GRADER_ENVIRONMENT = grading_environment(FIXTURE_GRADER_IMAGE)


def fixture_context(pipeline: RlDataPipeline, inputs: StagedInputs | None = None) -> ConversionContext:
    """The context the source pipeline supplies, with the fixture grader image for a declared recipe."""
    grader = FIXTURE_GRADER_ENVIRONMENT if pipeline.grader_image is not None else None
    return ConversionContext(inputs or {}, grader)


def convert_row(
    pipeline: RlDataPipeline, data: dict[str, Any], *, inputs: StagedInputs | None = None
) -> TaskSpec | NormalizedTask | ImportRejection:
    """Select, decode and convert one raw source row."""
    context = fixture_context(pipeline, inputs)
    recipe = source_recipe(pipeline, {}, context.grader_environment)
    if recipe.source.select is not None and not recipe.source.select(data, context):
        raise AssertionError(f"{pipeline.name} does not select its fixture row")
    decoded = recipe.source.decode(data, context) if recipe.source.decode is not None else data
    source = row_source(recipe, FIXTURE_LOCATOR)
    return pipeline.convert(RawRow(row_task_id(recipe, source), source, decoded), context)


def converted_task(pipeline: RlDataPipeline, data: dict[str, Any], *, inputs: StagedInputs | None = None) -> TaskSpec:
    """Convert a row that must produce a task whose agent environment matches the declaration."""
    result = convert_row(pipeline, data, inputs=inputs)
    if isinstance(result, ImportRejection):
        raise AssertionError(f"{pipeline.name} rejected its fixture row: {result}")
    task = result.task if isinstance(result, NormalizedTask) else result
    declared = pipeline.environment.requirements().docker_image
    assert task.environment_requirements.docker_image == declared, pipeline.name
    return task


def tasktrove_row(files: Mapping[str, bytes], *, path: str = "fixture-task") -> dict[str, Any]:
    """A TaskTrove ``tasks.parquet`` row whose archive holds ``files``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(data))
    return {"path": path, "task_binary": buffer.getvalue()}
