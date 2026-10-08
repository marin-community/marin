# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curation code and task-content identities."""

import hashlib
import sys
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

from verifyit.spec import Mode

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import SCHEMA_VERSION, NoGrader, ScriptGrader, TaskSpec, VerifyitGrader
from taskcompendium.pipeline.models import SourceRecipe

NORMALIZATION_STAGE_REVISION = "8"
VERIFICATION_STAGE_REVISION = "5"
REVIEW_STAGE_REVISION = "4"


def _value_identity(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if callable(value) and not isinstance(value, type):
        return callable_identity(value)
    if isinstance(value, dict):
        return {str(key): _value_identity(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_value_identity(item) for item in value]
    return value


def callable_identity(function: Callable[..., Any]) -> dict[str, Any]:
    """Name a function, a partial application, or a configured dataclass callable by code and arguments."""
    if isinstance(function, partial):
        return {
            "function": callable_identity(function.func),
            "args": [_value_identity(arg) for arg in function.args],
            "keywords": {key: _value_identity(value) for key, value in sorted(function.keywords.items())},
        }
    if is_dataclass(function) and not isinstance(function, type):
        kind = type(function)
        return {
            "module": kind.__module__,
            "name": kind.__qualname__,
            "parameters": {field.name: _value_identity(getattr(function, field.name)) for field in fields(function)},
        }
    return {"module": function.__module__, "name": function.__qualname__}


def function_code_identity(function: Callable[..., Any]) -> dict[str, Any]:
    """A callable's identity plus the digest of the module source that defines it."""
    module = sys.modules[callable_module(function)]
    assert module.__file__ is not None, "Code identity requires a module file"
    return {
        **callable_identity(function),
        "source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
    }


def callable_module(function: Callable[..., Any]) -> str:
    """The module defining a function, unwrapping partial applications and dataclass callables."""
    while isinstance(function, partial):
        function = function.func
    if is_dataclass(function) and not isinstance(function, type):
        return type(function).__module__
    return function.__module__


def recipe_code_identity(recipe: SourceRecipe) -> dict[str, str]:
    """Declare which code revisions can change this recipe's audit."""
    return {
        "normalization_stage": NORMALIZATION_STAGE_REVISION,
        "task_schema": SCHEMA_VERSION,
        "family": callable_module(recipe.convert),
        "family_revision": recipe.version,
        "verification_stage": VERIFICATION_STAGE_REVISION,
        "review_stage": REVIEW_STAGE_REVISION,
    }


def semantic_digest(task: TaskSpec, include_reference: bool) -> str:
    """Hash public task semantics, optionally including its private reference."""
    content = task.model_dump(mode="json", exclude={"id", "source"})
    # Oracle scripts are executable witnesses, not task semantics.
    content["resources"]["oracle"] = []
    grader = task.grader
    if include_reference and isinstance(grader, VerifyitGrader) and grader.mode in (Mode.MATH, Mode.MCQ):
        # These graders read only their parameters; derivations and provenance are audit evidence.
        content["resources"]["verifier"] = []
    if not include_reference:
        content.pop("grader")
        content["resources"]["verifier"] = []
    return canonical_sha256(content)


def deduplication_key(task: TaskSpec) -> str:
    """Opaque evaluator inputs and preference candidates define distinct task records."""
    # Different opaque contracts do not establish conflicting answer keys. Their
    # quality review checks reference agreement; exact copies still deduplicate.
    grader = task.grader
    opaque = isinstance(grader, ScriptGrader | NoGrader) or (
        isinstance(grader, VerifyitGrader) and grader.mode == Mode.SCRIPT
    )
    return semantic_digest(task, include_reference=opaque)
