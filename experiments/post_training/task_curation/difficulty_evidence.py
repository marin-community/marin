# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write immutable difficulty evidence shared by the coordinator and isolated worker."""

import hashlib
import json

from finestore.rl.curation import ContentEvidenceReference
from rigging.filesystem.storage_path import StoragePath


def write_json_evidence(path: StoragePath, value: dict) -> ContentEvidenceReference:
    if path.exists():
        raise FileExistsError(path)
    data = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(data)
    return ContentEvidenceReference(uri=str(path), sha256=hashlib.sha256(data).hexdigest())


def write_same_json_evidence(path: StoragePath, value: dict) -> ContentEvidenceReference:
    """Write immutable evidence, accepting a byte-identical prior write.

    A preempted root is retried with identical arguments and re-derives the same
    evidence (for example a task's pending receipt). An identical prior write is
    that same evidence, not a conflict; any different prior content is.
    """
    data = (json.dumps(value, indent=2) + "\n").encode()
    reference = ContentEvidenceReference(uri=str(path), sha256=hashlib.sha256(data).hexdigest())
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(path)
        return reference
    path.write_bytes(data)
    return reference
