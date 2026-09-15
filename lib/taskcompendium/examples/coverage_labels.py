# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the reviewed coverage labels for the bounded example suite."""

import json
from pathlib import Path

import msgspec

from taskcompendium.models import TaskSpec

_LABELS_PATH = Path(__file__).with_name("coverage-tags.json")


def label(specification: TaskSpec) -> TaskSpec:
    """Attach the reviewed coverage tags for one bounded-suite specification."""
    tags = json.loads(_LABELS_PATH.read_text())[specification.id]
    return msgspec.structs.replace(specification, coverage_tags=tuple(tags))
