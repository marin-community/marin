# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Served status rows render only from the canonical enums.

The serve-time guard canonicalizes each declaration's typed fields against the
taxonomy; these tests pin the contract the status page and the source table
depend on: members pass unchanged, legacy-free drift fails loudly.
"""

import pytest
from finestore.rl.status_taxonomy import PendingStage, SourceStatus

from infra.marina.apps.rl_data_catalog.server.typed_status import require_typed_served_declaration


def test_canonical_members_pass_through_unchanged():
    row = {
        "id": "Fixture:typed",
        "status": "Pending",
        "stage": "Intake queued",
        "intended_use": "train",
        "method_tags": ["RL", "SFT-viable"],
        "exclusion_claims": [],
    }
    assert require_typed_served_declaration(row) == row


def test_out_of_enum_status_fails_loudly():
    with pytest.raises(ValueError, match="outside SourceStatus"):
        require_typed_served_declaration({"id": "Fixture:drift", "status": "Not selectable"})


def test_out_of_enum_fields_fail_with_the_source_id():
    for field, value in [
        ("stage", "Sometime soon"),
        ("intended_use", "distillation"),
        ("method_tags", ["legacy"]),
    ]:
        with pytest.raises(ValueError, match="Fixture:drift"):
            require_typed_served_declaration({"id": "Fixture:drift", field: value})


def test_out_of_enum_claim_reason_fails():
    claim = {"reason": "needs-a-ruling-someday", "evidence": {"kind": "note", "message": "x"}}
    with pytest.raises(ValueError, match="outside ExclusionReason"):
        require_typed_served_declaration({"id": "Fixture:drift", "exclusion_claims": [claim]})


def test_absent_typed_fields_pass():
    row = require_typed_served_declaration({"id": "Fixture:sparse", "notes": "declaration only"})
    assert "status" not in row
    assert row["exclusion_claims"] == []


def test_status_members_carry_the_served_labels():
    for status in SourceStatus:
        assert status.value
    assert SourceStatus.PENDING.value == "Pending"
    assert PendingStage.INTAKE_QUEUED.value == "Intake queued"
