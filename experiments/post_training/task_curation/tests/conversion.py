# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert fixture rows through a declaration the way the source pipeline does."""

import io
import tarfile
from collections.abc import Mapping
from typing import Any

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.conversion import row_source, row_task_id
from taskcompendium.pipeline.inputs import ConversionContext, StagedInputs
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask, RawRow

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.environment import Environment, Placement, placement
from experiments.post_training.task_curation.images.build import BASE_IMAGE, PYTHON_VERSION, EnvironmentArtifact
from experiments.post_training.task_curation.pipeline import (
    RlDataPipeline,
    ShellSim,
    environment_requirements,
    source_recipe,
)

FIXTURE_LOCATOR = "fixture.jsonl:0"
FIXTURE_GRADER_IMAGE = "ghcr.io/marin-community/task-curation-grader@sha256:" + "0" * 64
FIXTURE_ENVIRONMENT_PATH = "/fixture/images/env-0000000000000000"


def fixture_build(environment: Environment) -> EnvironmentArtifact:
    """Stands in for an environment's built artifact: conversion records its lock or image without using it."""
    return EnvironmentArtifact(
        path=FIXTURE_ENVIRONMENT_PATH,
        identity="0" * 64,
        lock_sha256="0" * 64,
        apt=list(environment.apt),
        data=list(environment.data),
        python=PYTHON_VERSION,
        base_image=BASE_IMAGE,
        image=FIXTURE_GRADER_IMAGE if placement(environment) == Placement.BUILT_IMAGE else None,
    )


FIXTURE_GRADER_ENVIRONMENT = environment_requirements(GRADER_PACKAGES, fixture_build(GRADER_PACKAGES))
"""The grader packages' environment as the source pipeline supplies it to converters."""


def fixture_context(pipeline: RlDataPipeline, inputs: StagedInputs | None = None) -> ConversionContext:
    """The context the source pipeline supplies, with a fixture build of the declared grader environment."""
    grader = None
    if pipeline.grader is not None:
        built = fixture_build(pipeline.grader) if placement(pipeline.grader) != Placement.IMAGE else None
        grader = environment_requirements(pipeline.grader, built)
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
    declared = None if isinstance(pipeline.environment, ShellSim) else pipeline.environment.image
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
