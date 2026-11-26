# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Order-independent JSON digests shared by the served applet and the release tooling."""

import hashlib
import json
from typing import Any


def json_sha256(value: Any) -> str:
    """Hash JSON independently of object key order and whitespace."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
