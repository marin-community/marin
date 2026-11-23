# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepared source contracts shared with Harbor's isolated trial process."""

import base64
from collections.abc import Mapping

from finestore.rl.curation import ContentEvidenceReference
from finestore.rl.difficulty import DifficultySample
from pydantic import BaseModel, ConfigDict, Field

GRADER_PRIVILEGE_DROP_UID = "user=65534"


def grader_requires_multi_uid(spec: Mapping) -> bool:
    """Check whether the frozen verifier requests a fixed-UID privilege drop.

    LOCAL maps one UID, so these graders need Daytona's full UID map.
    """
    for resource in (spec.get("resources") or {}).get("verifier") or ():
        content = base64.b64decode(resource["source"]["content_base64"]).decode("utf-8", errors="replace")
        if GRADER_PRIVILEGE_DROP_UID in content:
            return True
    return False


class PreparedTask(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: str | None
    workload: str
    backend: str
    reason: str | None
    judge_model: str | None


class PreparedSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    sample: DifficultySample
    sample_receipt: ContentEvidenceReference
    tasks: list[PreparedTask]
