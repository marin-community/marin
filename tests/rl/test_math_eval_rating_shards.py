# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from experiments.post_training.math_eval.rating_shards import METADATA_KEYS, PROTOCOL_KEYS, RUNTIME_KEYS, _sha
from experiments.post_training.math_eval.rating_shards import merge_rating_shards as merge_pinned


def merge_rating_shards(shards, *, expected_ids):
    return merge_pinned(
        shards, expected_ids=expected_ids, expected_shard_sha256=[pair[0]["ratings_sha256"] for pair in shards]
    )


def shard(digest):
    protocol = {key: "stable" for key in PROTOCOL_KEYS}
    protocol["runtime"] = {key: "stable" for key in RUNTIME_KEYS}
    protocol["run_id"] = digest
    audit = {
        "schema": "math_eval_serving_audit_v1",
        "inference_evidence_pass": True,
        "clean_end_to_end": True,
        "protocol": protocol,
        "records_sha256": digest,
        "expected_ids_sha256": _sha([digest]),
    }
    metadata = {key: "stable" for key in METADATA_KEYS}
    metadata.update(
        samples=8,
        generation_protocol_verified=True,
        generation_provenance=protocol,
        records_sha256=digest,
        generation_audit_sha256=_sha(audit),
    )
    overlay = {
        "metadata": metadata,
        "ratings": [{"prompt_sha256": digest, "samples": 8, "passes": 3, "pass_rate_k": 3 / 8}],
    }
    return overlay | {"ratings_sha256": _sha(overlay)}, audit


def rehash(pair):
    overlay, audit = pair
    overlay["metadata"]["generation_audit_sha256"] = _sha(audit)
    overlay["ratings_sha256"] = _sha({key: value for key, value in overlay.items() if key != "ratings_sha256"})
    return pair


def test_disjoint_rating_shards_retain_distinct_native_provenance_and_order_independence():
    shards = [shard("a" * 64), shard("b" * 64)]
    original = deepcopy(shards)
    merged = merge_rating_shards(shards, expected_ids=["b" * 64, "a" * 64])
    assert merged == merge_rating_shards(shards[::-1], expected_ids=["a" * 64, "b" * 64])
    assert shards == original
    assert len(merged["source_shards"]) == 2
    assert "records_sha256" not in merged["metadata"]
    assert [row["passes"] for row in merged["ratings"]] == [3, 3]


@pytest.mark.parametrize("poison", ["duplicate", "missing", "sampling", "unverified", "records", "hash", "sample_count"])
def test_rating_shards_reject_duplication_or_unproven_incompatible_inputs(poison):
    shards = [shard("a" * 64), shard("b" * 64)]
    if poison == "duplicate":
        shards[1] = deepcopy(shards[0])
    elif poison == "missing":
        shards.pop()
    elif poison == "sampling":
        shards[1][1]["protocol"]["temperature"] = 0.6
        rehash(shards[1])
    elif poison == "unverified":
        shards[1][1]["inference_evidence_pass"] = False
        rehash(shards[1])
    elif poison == "records":
        shards[1][1]["records_sha256"] = "wrong"
        rehash(shards[1])
    elif poison == "hash":
        shards[1][0]["ratings_sha256"] = "wrong"
    elif poison == "sample_count":
        shards[1][0]["ratings"][0]["samples"] = 4
        rehash(shards[1])
    with pytest.raises(ValueError):
        merge_rating_shards(shards, expected_ids=["a" * 64, "b" * 64])


def test_modified_overlay_cannot_replace_a_previously_pinned_shard_receipt():
    pair = shard("a" * 64)
    pinned = pair[0]["ratings_sha256"]
    pair[0]["ratings"][0].update(passes=4, pass_rate_k=0.5)
    rehash(pair)
    with pytest.raises(ValueError, match="immutable generation audit"):
        merge_pinned([pair], expected_ids=["a" * 64], expected_shard_sha256=[pinned])
