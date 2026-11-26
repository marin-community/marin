# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TB2 recovery stitch's identity and replacement rules."""

from experiments.post_training.mochi_loop_guards.tb2_recovery import digest, row_key, stitched_rows


def test_digest_is_canonical_over_key_order():
    assert digest({"a": 1, "b": [2, 3]}) == digest({"b": [2, 3], "a": 1})
    assert digest({"a": 1}) != digest({"a": "1"})


def test_row_key_binds_prompt_content_and_outcome_fields():
    saved = {
        "uid": "u0",
        "response_token_ids": [1, 2],
        "messages": [{"role": "user", "content": "hi"}],
        "verification": {"status": "verified"},
        "exception_type": None,
    }
    assert row_key(saved) == row_key(dict(saved))
    replay_failed = {**saved, "verification": {"status": "failed"}}
    assert row_key(saved) != row_key(replay_failed)
    different_content = {**saved, "response_token_ids": [1, 3]}
    assert row_key(saved) != row_key(different_content)


def original(task, repetition, ordinal, outcome):
    return {"task_id": task, "repetition": repetition, "ordinal": ordinal, "outcome": outcome}


def test_stitched_rows_replaces_only_target_identities_and_records_provenance():
    originals = {
        ("tb2/a", 0): (original("tb2/a", 0, 4, "graded"), {"record_id": "rec-0"}),
        ("tb2/a", 1): (original("tb2/a", 1, 7, "infrastructure"), {"record_id": "rec-1"}),
    }
    replays = {("tb2/a", 1): (original("tb2/a", 1, 0, "graded"), "rec-replay", "archive.zip")}
    rows, replacements = stitched_rows(originals, replays, {("tb2/a", 1)}, sampling_seed=lambda task, repetition: 99)

    # Sorted by identity; the graded original is preserved verbatim.
    assert [row["repetition"] for row in rows] == [0, 1]
    graded, replaced = rows
    assert graded["outcome"] == "graded"
    assert replaced["outcome"] == "graded"
    assert replaced["recovery_provenance"] == {
        "task_id": "tb2/a",
        "repetition_id": 1,
        "sampling_seed": 99,
        "original_record_id": "rec-1",
        "original_dump_ordinal": 7,
        "replaced_original_infrastructure": True,
        "replay_record_id": "rec-replay",
        "replay_archive": "archive.zip",
    }
    # The replay's verification status rides in the replacement records, not the rows.
    assert "replay_verification_status" not in replaced["recovery_provenance"]
    assert graded["recovery_provenance"] == {
        "task_id": "tb2/a",
        "repetition_id": 0,
        "sampling_seed": 99,
        "original_record_id": "rec-0",
        "original_dump_ordinal": 4,
    }
    assert len(replacements) == 1
    assert replacements[0]["replay_verification_status"] is None
    # The stitched rows are new dicts; the saved originals are never mutated.
    assert rows[0] is not originals[("tb2/a", 0)][0]
