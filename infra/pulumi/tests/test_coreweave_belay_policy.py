# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iac.buckets.coreweave import BELAY_PRINCIPAL, BELAY_RUN_PREFIXES, belay_policy_statements

WRITE_ACTIONS = {"s3:PutObject", "s3:DeleteObject"}


def test_belay_writes_only_its_own_run_prefixes() -> None:
    writes = [s for s in belay_policy_statements() if WRITE_ACTIONS & set(s.actions)]
    assert writes
    assert all(set(s.resources) <= set(BELAY_RUN_PREFIXES) for s in writes)
    assert all("/belay/" in prefix for prefix in BELAY_RUN_PREFIXES)


def test_belay_object_access_is_prefix_scoped() -> None:
    for statement in belay_policy_statements():
        assert statement.principals == [BELAY_PRINCIPAL]
        assert "s3:*" not in statement.actions
        if statement.actions != ["s3:ListBucket"]:
            assert all(resource.count("/") >= 2 for resource in statement.resources)
