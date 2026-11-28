# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The reviewed converter module supplies format and grader-environment policy."""

from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

from finestore.rl.curation import TaskFormat
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import Converter

from experiments.post_training.task_curation.environment import Environment


@dataclass(frozen=True)
class RowConverterBinding:
    convert: Converter
    source_format: SourceFormat
    task_format: TaskFormat
    grader: Environment | None


class ConverterModule(Protocol):
    INGEST_CONVERTER: RowConverterBinding


def load_converter(module: str) -> RowConverterBinding:
    """Load the explicit maintained binding; missing bindings have no fallback."""
    return cast(ConverterModule, import_module(module)).INGEST_CONVERTER
