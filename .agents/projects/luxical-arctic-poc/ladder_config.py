# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed configuration for the Arctic-distilled Luxical ladder."""

from enum import StrEnum

SEED = 42
OUTPUT_ROOT = "s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder"
SOURCE_INVENTORY_URL = f"{OUTPUT_ROOT}/source_inventory.json"
MANIFEST_ROOT = f"{OUTPUT_ROOT}/manifest-v1"
TRAIN_TARGET_750K = 750_000
TRAIN_TARGET_3M = 3_000_000
EVAL_ROWS_PER_SOURCE = 512
SURVEY_ROWS_PER_SOURCE = 100
MAX_TRAIN_ROWS_PER_SOURCE = 50_000
MIN_SOURCES = 140
TEXT_WINDOW_CHARS = 2_000
TEACHER_ID = "Snowflake/snowflake-arctic-embed-m-v2.0"
TEACHER_REVISION = "95c2741480856aa9666782eb4afe11959938017f"

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
