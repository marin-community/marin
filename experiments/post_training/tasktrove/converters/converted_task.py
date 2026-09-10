# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What a converter takes and returns, and how converters are keyed."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from tasktrove_verify.spec import Spec

from experiments.post_training.tasktrove.taskbinary import TaskFiles


class ConvertStatus(StrEnum):
    CONVERTED = "converted"
    NO_CONVERTER = "no_converter"
    CONVERTER_ERROR = "converter_error"
    DROPPED_SOURCE = "dropped_source"
    # Typed rejections a converter returns for a task its template cannot grade soundly.
    NULL_GRADER = "null_grader"
    TOO_FEW_CASES = "too_few_cases"
    GOLD_IN_INSTRUCTION = "gold_in_instruction"
    UNSUPPORTED_VARIANT = "unsupported_variant"


@dataclass(frozen=True)
class ConvertedTask:
    instruction: str
    spec: Spec
    dockerfile: str
    """The task's own Dockerfile after the converter's edits; the pipeline appends the tool install."""
    tags: tuple[str, ...]
    language: str = ""
    data_files: dict[str, bytes] = field(default_factory=dict)
    """Files the spec references, keyed by path under the task root (``tests/cases/...``)."""
    solution_files: dict[str, bytes] = field(default_factory=dict)
    """Oracle solution; stored beside the task, never inside the binary the agent sees."""
    metadata: dict = field(default_factory=dict)
    """Converter-specific ``task.toml`` metadata; the template's own ``metadata.json`` is merged by ``convert_one``."""
    agent_timeout: float = 900.0
    verifier_timeout: float = 600.0


@dataclass(frozen=True)
class Rejected:
    status: ConvertStatus
    detail: str


@dataclass(frozen=True)
class ConverterKey:
    """What selects a converter: the source family from the verdicts and the template's code files."""

    family: str
    code_files: frozenset[str]


ConvertFn = Callable[[TaskFiles], ConvertedTask | Rejected]


@dataclass(frozen=True)
class Converter:
    name: str
    keys: tuple[ConverterKey, ...]
    convert: ConvertFn
