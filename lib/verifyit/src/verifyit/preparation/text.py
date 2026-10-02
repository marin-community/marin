# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lossless text capture followed by named, potentially grade-affecting policies."""

import re
import string
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, cast

from harbor_config.errors import error_category

from verifyit.grade import InvalidTask, Status, finalize_preparation_failure
from verifyit.preparation.errors import InvalidPreparation, PreparationError, PreparationFailure


class TextPolicy(StrEnum):
    HARNESS_EXACT = "harness_exact_v1"
    IDENTITY = "identity"


@dataclass(frozen=True)
class TextInputs:
    candidate: str
    references: tuple[str, ...]


@dataclass(frozen=True)
class TextNormalization:
    policy: TextPolicy = TextPolicy.HARNESS_EXACT
    regexes_to_ignore: tuple[str, ...] = ()
    ignore_case: bool = False
    ignore_punctuation: bool = False
    ignore_numbers: bool = False


@dataclass(frozen=True)
class PreparedText:
    raw: TextInputs
    candidate: str
    references: tuple[str, ...]
    normalization: TextNormalization


def _preparation_error(error: Exception, status: Status, stage: str) -> PreparationError:
    failure = PreparationFailure(status, error_category(type(error).__name__), type(error).__name__, str(error), stage)
    verdict = finalize_preparation_failure(**asdict(failure))
    exception = InvalidPreparation if status is Status.INVALID_TASK else PreparationError
    return exception(failure, verdict)


def structure_text(candidate: str, references: Sequence[str]) -> TextInputs:
    """Snapshot strings without filtering, coercion, reordering, or normalization."""
    if not references or any(not isinstance(reference, str) for reference in references):
        raise _preparation_error(
            InvalidTask("exact_match references must be a nonempty sequence of strings"),
            Status.INVALID_TASK,
            "structure",
        )
    if not isinstance(candidate, str):
        raise _preparation_error(InvalidTask("exact_match candidate must be a string"), Status.INVALID_TASK, "structure")
    return TextInputs(candidate, tuple(references))


def normalize_text(inputs: TextInputs, normalization: TextNormalization) -> PreparedText:
    """Apply the selected policy, retaining its input snapshot and effective options.

    Harness policy deliberately uses separate fixed-width NumPy arrays for the
    candidate and references. Combining their batches changes Unicode lowering.
    """
    if not isinstance(normalization.policy, TextPolicy) or any(
        not isinstance(flag, bool)
        for flag in (normalization.ignore_case, normalization.ignore_punctuation, normalization.ignore_numbers)
    ):
        raise _preparation_error(
            InvalidTask("invalid text normalization policy or flags"), Status.INVALID_TASK, "normalize"
        )
    if not isinstance(normalization.regexes_to_ignore, tuple) or any(
        not isinstance(pattern, str) for pattern in normalization.regexes_to_ignore
    ):
        raise _preparation_error(
            InvalidTask("normalization regexes must be a tuple of strings"), Status.INVALID_TASK, "normalize"
        )
    if normalization.policy == TextPolicy.IDENTITY:
        if normalization.regexes_to_ignore or any(
            (normalization.ignore_case, normalization.ignore_punctuation, normalization.ignore_numbers)
        ):
            raise _preparation_error(
                InvalidTask("identity policy does not accept normalization options"), Status.INVALID_TASK, "normalize"
            )
        return PreparedText(inputs, inputs.candidate, inputs.references, normalization)
    try:
        patterns = tuple(re.compile(pattern) for pattern in normalization.regexes_to_ignore)
    except re.error as error:
        raise _preparation_error(error, Status.INVALID_TASK, "normalize") from error
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as error:
        raise _preparation_error(error, Status.INFRA_ERROR, "normalize") from error
    batches = []
    for values in ((inputs.candidate,), inputs.references):
        for pattern in patterns:
            values = tuple(pattern.sub("", value) for value in values)
        array = np.asarray(values)
        if normalization.ignore_case:
            array = np.char.lower(array)
        for enabled, characters in (
            (normalization.ignore_punctuation, string.punctuation),
            (normalization.ignore_numbers, string.digits),
        ):
            if enabled:
                array = np.char.translate(array, table=cast(Any, str.maketrans("", "", characters)))
        batches.append(tuple(array.tolist()))
    return PreparedText(inputs, batches[0][0], batches[1], normalization)
