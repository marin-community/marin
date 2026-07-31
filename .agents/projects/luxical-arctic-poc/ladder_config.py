# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed configuration for the Arctic-distilled Luxical ladder."""

from enum import StrEnum

SEED = 42
OUTPUT_ROOT = "s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder"
SOURCE_INVENTORY_URL = f"{OUTPUT_ROOT}/source_inventory.json"
MANIFEST_ROOT = f"{OUTPUT_ROOT}/manifest-v2"
TRAIN_TARGET_750K = 750_000
TRAIN_TARGET_3M = 3_000_000
EVAL_ROWS_PER_SOURCE = 512
SURVEY_ROWS_PER_SOURCE = 100
MAX_TRAIN_ROWS_PER_SOURCE = 50_000
MIN_SOURCES = 140
SAMPLE_BLOCKS_PER_SOURCE = 64
SAMPLING_METHOD = "uniform_marginal_circular_blocks_without_duplicate_positions"
TEXT_WINDOW_CHARS = 2_000
TEACHER_ID = "Snowflake/snowflake-arctic-embed-m-v2.0"
TEACHER_REVISION = "95c2741480856aa9666782eb4afe11959938017f"
STACK_V3_OUTPUT_HASH = "32b6fa6f"

PREDECLARED_OOD_SOURCES = frozenset(
    (
        "ghalogs/public",
        "massive_function_calling",
        "molmo2-cap",
        "svg",
    )
)

CODE_SOURCE_MARKERS = (
    "agenttrove",
    "code",
    "coderforge",
    "davinci-dev/",
    "kernelgym",
    "nemotron-terminal",
    "stack-v3",
    "starcoder2/",
    "swe-",
)

MULTILINGUAL_SOURCE_MARKERS = (
    "climblab-ja",
    "finepdfs/",
    "finetranslations",
    "translated",
)


class SourceCategory(StrEnum):
    """Define the source groups used by the fixed evaluation."""

    CODE = "code"
    MULTILINGUAL = "multilingual"
    OOD = "ood"
    STANDARD = "standard"


def source_category(source: str) -> SourceCategory:
    """Return the fixed evaluation category for one source."""
    if source in PREDECLARED_OOD_SOURCES:
        return SourceCategory.OOD
    if any(marker in source for marker in CODE_SOURCE_MARKERS):
        return SourceCategory.CODE
    if any(marker in source for marker in MULTILINGUAL_SOURCE_MARKERS):
        return SourceCategory.MULTILINGUAL
    return SourceCategory.STANDARD


def document_windows(text: str) -> tuple[str, str, str]:
    """Return the head, middle, and tail windows of one document."""
    middle_start = max(0, len(text) // 2 - TEXT_WINDOW_CHARS // 2)
    return (
        text[:TEXT_WINDOW_CHARS],
        text[middle_start : middle_start + TEXT_WINDOW_CHARS],
        text[-TEXT_WINDOW_CHARS:],
    )


def document_view(text: str) -> str:
    """Return the bounded student view of one document."""
    if len(text) <= 3 * TEXT_WINDOW_CHARS:
        return text
    return "\n".join(document_windows(text))


def teacher_windows_from_view(text: str) -> tuple[str, str, str]:
    """Return the teacher windows encoded in a student view."""
    long_view_characters = 3 * TEXT_WINDOW_CHARS + 2
    if len(text) != long_view_characters:
        return document_windows(text)
    first_separator = TEXT_WINDOW_CHARS
    second_separator = 2 * TEXT_WINDOW_CHARS + 1
    if text[first_separator] != "\n" or text[second_separator] != "\n":
        raise ValueError("A long document view has invalid window separators")
    return (
        text[:first_separator],
        text[first_separator + 1 : second_separator],
        text[second_separator + 1 :],
    )
