# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The declaration audit pairs every Atlas receipt family with the canonical layer."""

from experiments.post_training.task_curation.declaration_audit import (
    CanonicalLayer,
    DeclaredPin,
    ReleasePin,
    audit_families,
)


def pin(source_id, dataset_id="open-athena/example-source", revision="1" * 40, version="2026.11.13.1"):
    return DeclaredPin(source_id=source_id, dataset_id=dataset_id, dataset_revision=revision, converter_version=version)


def release(repo="open-athena/example-release", revision="2" * 40):
    return ReleasePin(repo=repo, revision=revision)


def layer(**overrides) -> CanonicalLayer:
    base = dict(
        source_ids=frozenset({"Test:declared-source"}),
        declared={"Test:declared-source": pin("Test:declared-source")},
        selected={"Test:declared-source": release()},
        retired=frozenset(),
        archived_releases={},
    )
    base.update(overrides)
    return CanonicalLayer(**base)


def test_a_receipt_at_the_declared_pin_binds():
    findings, notes = audit_families(
        {
            "census receipts": {"Test:declared-source": pin("Test:declared-source")},
            "within-source dedup": {"Test:declared-source": pin("Test:declared-source")},
            "lexical receipts": {"Test:declared-source": pin("Test:declared-source")},
        },
        layer=layer(),
        releases={"Test:declared-source": release()},
    )
    assert findings == []
    assert notes == []


def test_a_registry_only_receipt_fails():
    findings, _ = audit_families(
        {"census receipts": {"Test:undeclared-source": pin("Test:undeclared-source")}},
        layer=layer(),
    )
    assert findings == [
        "census receipts: Test:undeclared-source names a source the declaration layer does not declare "
        "(registry-only receipt at " + "1" * 40 + ")"
    ]


def test_a_stale_declaration_pin_fails():
    findings, _ = audit_families(
        {
            "census receipts": {"Test:declared-source": pin("Test:declared-source", revision="9" * 40)},
            "within-source dedup": {"Test:declared-source": pin("Test:declared-source", version="2026.11.12.1")},
        },
        layer=layer(),
    )
    assert len(findings) == 2
    assert all("but the canonical layer declares" in finding for finding in findings)


def test_a_lexical_receipt_at_neither_declared_nor_selected_pin_fails():
    findings, _ = audit_families(
        {"lexical receipts": {"Test:declared-source": pin("Test:declared-source", revision="8" * 40)}},
        layer=layer(),
    )
    assert findings == [
        "lexical receipts: Test:declared-source receipt pin open-athena/example-source@" + "8" * 40
        + " (2026.11.13.1) matches no declared or selected pin"
    ]


def test_a_retained_receipt_must_match_the_selected_release():
    findings, _ = audit_families(
        {},
        layer=layer(),
        releases={"Test:declared-source": release(repo="open-athena/other-release")},
    )
    assert findings == [
        "retained release receipts: Test:declared-source selects open-athena/example-release@"
        + "2" * 40
        + " but its receipt publishes open-athena/other-release@" + "2" * 40
    ]


def test_a_receipt_for_a_retired_source_is_history_not_drift():
    retired_layer = layer(source_ids=frozenset(), retired=frozenset({"Test:retired-source"}))
    findings, _ = audit_families(
        {
            "census receipts": {"Test:retired-source": pin("Test:retired-source", revision="9" * 40)},
            "within-source dedup": {"Test:retired-source": pin("Test:retired-source")},
        },
        layer=retired_layer,
        releases={"Test:retired-source": release()},
    )
    assert findings == []


def test_a_filtration_superseded_retained_receipt_binds():
    superseded = layer(
        source_ids=frozenset({"Test:declared-source", "Test:filtered-source"}),
        declared={
            "Test:declared-source": pin("Test:declared-source"),
            "Test:filtered-source": pin("Test:filtered-source"),
        },
        selected={
            "Test:declared-source": release(),
            "Test:filtered-source": release(repo="open-athena/filtered-release", revision="3" * 40),
        },
        superseded_by_filtration=frozenset({"Test:filtered-source"}),
    )
    findings, notes = audit_families(
        {},
        layer=superseded,
        releases={"Test:filtered-source": release(revision="4" * 40)},
    )
    assert findings == []
    assert notes == []


def test_a_pending_source_retained_release_is_reported_staged_for_intake():
    pending = layer(
        source_ids=frozenset({"Test:declared-source", "Test:pending-source"}),
        declared={
            "Test:declared-source": pin("Test:declared-source"),
            "Test:pending-source": pin("Test:pending-source"),
        },
        superseded_by_filtration=frozenset(),
        pending=frozenset({"Test:pending-source"}),
    )
    findings, notes = audit_families(
        {
            "census receipts": {"Test:pending-source": pin("Test:pending-source")},
            "within-source dedup": {"Test:pending-source": pin("Test:pending-source")},
        },
        layer=pending,
        releases={"Test:pending-source": release(repo="open-athena/pending-remediated-release")},
    )
    assert findings == []
    assert notes == [
        "retained release receipts: Test:pending-source is pending intake; its receipt for "
        "open-athena/pending-remediated-release@" + "2" * 40 + " stays staged until adoption selects it"
    ]


def test_an_archived_original_record_counts_as_declared():
    archived = layer(
        source_ids=frozenset({"Test:archived-source"}),
        declared={
            "Test:archived-source": pin("Test:archived-source", dataset_id="open-athena/original-conversion"),
        },
        selected={"Test:archived-source": release(repo="open-athena/admitted-release", revision="5" * 40)},
        archived_releases={"Test:archived-source": release(repo="open-athena/archived-retained", revision="6" * 40)},
    )
    findings, notes = audit_families(
        {"census receipts": {"Test:archived-source": pin("Test:archived-source", dataset_id="open-athena/original-conversion")}},
        layer=archived,
        releases={"Test:archived-source": release(repo="open-athena/archived-retained", revision="6" * 40)},
    )
    assert findings == []
    assert notes == []
