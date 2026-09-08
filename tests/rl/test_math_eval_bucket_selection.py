# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from experiments.post_training.math_eval.audit_overlay import VERIFIER_REVISION, VERIFIER_SOURCES_SHA256
from experiments.post_training.math_eval.bucket_selection import completed_group_statistics, select_qwen_bucket_s
from experiments.post_training.math_eval.rating_shards import METADATA_KEYS, PROTOCOL_KEYS, RUNTIME_KEYS, _sha


def fixture(counts=(0, 2, 4, 6, 8)):
    manifest = [{"prompt_sha256": f"{i:064x}", "bin": "fixture", "split": "train"} for i in range(len(counts))]
    manifest.append({"prompt_sha256": "f" * 64, "bin": "fixed", "split": "heldout"})
    manifest_sha = _sha(manifest)
    mechanical = {
        "manifest_sha256": manifest_sha,
        "verifier_revision": VERIFIER_REVISION,
        "verifier_sources_sha256": VERIFIER_SOURCES_SHA256,
        "audit_source_sha256": "a" * 64,
        "statuses": {row["prompt_sha256"]: "accept" for row in manifest},
    }
    ids = [row["prompt_sha256"] for row in manifest[:-1]]
    protocol = {key: "stable" for key in PROTOCOL_KEYS}
    protocol["runtime"] = {key: "stable" for key in RUNTIME_KEYS}
    generation = {
        "schema": "math_eval_serving_audit_v1",
        "inference_evidence_pass": True,
        "clean_end_to_end": True,
        "protocol": protocol,
        "records_sha256": "b" * 64,
        "expected_ids_sha256": _sha(ids),
    }
    metadata = {key: "stable" for key in METADATA_KEYS}
    metadata.update(
        model="qwen",
        samples=8,
        metric="score_contract_completed",
        manifest_sha256=manifest_sha,
        audit_overlay_sha256=_sha(mechanical),
        generation_protocol_verified=True,
        generation_provenance=protocol,
        records_sha256="b" * 64,
        generation_audit_sha256=_sha(generation),
    )
    ratings = {
        "metadata": metadata,
        "ratings": [
            {"prompt_sha256": uid, "samples": 8, "passes": k, "pass_rate_k": k / 8}
            for uid, k in zip(ids, counts, strict=True)
        ],
    }
    ratings["ratings_sha256"] = _sha(ratings)
    return manifest, {"manifest_sha256": manifest_sha}, mechanical, ratings, generation


def select(args):
    return select_qwen_bucket_s(*args, expected_ratings_sha256=args[3]["ratings_sha256"])


def test_selection_preserves_all_bins_and_excludes_extremes_and_fixed_heldout():
    args = fixture()
    original = deepcopy(args)
    result = select(args)
    assert result["adoptable"]
    assert result["prompt_sha256"] == [f"{i:064x}" for i in (1, 2, 3)]
    assert result["combined"]["completed_pass1"] == 0.5
    assert result["all_source_bins"]["fixture"]["excluded_zero_of_eight"] == 1
    assert result["all_source_bins"]["fixture"]["excluded_eight_of_eight"] == 1
    assert args == original


def test_full_bin_can_pass_while_actual_eligible_view_fails():
    result = select(fixture((0, 6, 6)))
    assert result["all_source_bins"]["fixture"]["full_source"]["meets_thresholds"]
    assert result["combined"]["completed_pass1"] == 0.75
    assert not result["adoptable"]
    assert completed_group_statistics([])["questions"] == 0


@pytest.mark.parametrize("poison", ["audit", "missing", "hash", "manifest", "reject"])
def test_selection_refuses_unproven_or_incomplete_rating_membership(poison):
    args = list(fixture())
    if poison == "audit":
        args[4]["clean_end_to_end"] = False
    elif poison == "missing":
        args[3]["ratings"].pop()
    elif poison == "hash":
        args[3]["ratings_sha256"] = "0" * 64
    elif poison == "manifest":
        args[0][0]["bin"] = "foreign"
    elif poison == "reject":
        args[2]["statuses"][f"{0:064x}"] = "reject"
    with pytest.raises(ValueError):
        select(args)
