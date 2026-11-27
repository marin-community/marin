# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import replace

import pytest

from experiments.evaluation.micro210_policy import LEGACY_POLICY_REFERENCE, POLICY_REFERENCE, read_micro210_policy
from tests.rl.micro210_fixture import policy_fixture_reference


def test_default_policy_references_pin_immutable_hugging_face_revisions():
    for reference in (LEGACY_POLICY_REFERENCE, POLICY_REFERENCE):
        assert re.fullmatch(r"hf://datasets/open-athena/micro210-policy@[0-9a-f]{40}/policy.json", reference.uri)


def test_micro210_policy_rejects_modified_protocol_bytes(tmp_path):
    reference = policy_fixture_reference()
    assert read_micro210_policy(reference)["evaluation"]["seed"] > 0
    changed = tmp_path / "policy.json"
    with open(reference.uri, "rb") as original:
        changed.write_bytes(original.read().replace(b"20261030", b"20261031"))
    with pytest.raises(ValueError, match="hash mismatch"):
        read_micro210_policy(replace(reference, uri=str(changed)))
