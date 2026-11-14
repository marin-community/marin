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
