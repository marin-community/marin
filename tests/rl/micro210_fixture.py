# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small offline protocol fixture for the Micro210 launch boundary."""

import hashlib
from pathlib import Path

from experiments.evaluation.micro210_policy import Micro210PolicyReference


def policy_fixture_reference() -> Micro210PolicyReference:
    path = Path(__file__).with_name("fixtures") / "synthetic-micro210-policy.json"
    raw = path.read_bytes()
    return Micro210PolicyReference(str(path), "fixture", hashlib.sha256(raw).hexdigest(), len(raw))
