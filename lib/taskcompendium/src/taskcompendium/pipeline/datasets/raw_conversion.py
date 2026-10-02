# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose a source converter with a recipe's typed normalization."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    ImportRejection,
    NormalizationChange,
    NormalizedTask,
    RawRow,
)

RawConverter = Callable[[Mapping[str, Any]], dict[str, Any]]


def with_raw_converter(recipe: DatasetRecipe, converter: RawConverter, revision: str) -> DatasetRecipe:
    """Record converter repairs and identity beside the source's normalizer."""
    normalize = recipe.normalize

    def normalize_raw(row: RawRow) -> TaskSpec | NormalizedTask | ImportRejection:
        prepared = converter(row.data)
        result = normalize(RawRow(row.id, row.source, prepared))
        if isinstance(result, ImportRejection):
            return result
        changes = tuple(
            NormalizationChange.model_validate(change) for change in prepared["converted"]["normalization_changes"]
        )
        if isinstance(result, NormalizedTask):
            return NormalizedTask(result.task, changes + result.changes)
        return NormalizedTask(result, changes)

    suite = recipe.check_suite
    assert suite is not None
    return replace(
        recipe,
        version=f"{recipe.version}-raw-conversion-v1",
        normalize=normalize_raw,
        check_suite=replace(suite, parameters={**suite.parameters, "converter_revision": revision}),
    )
