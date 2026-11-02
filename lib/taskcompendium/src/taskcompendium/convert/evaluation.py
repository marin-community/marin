# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small evaluation converters that preserve benchmark prompts and private source evidence."""

import json
from typing import Any

from taskcompendium.convert.answers import EVIDENCE_PATH
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes


def source_evidence(task: TaskSpec) -> dict[str, Any]:
    resource = next(resource for resource in task.resources.verifier if resource.path == EVIDENCE_PATH)
    return json.loads(resource_bytes(resource))
