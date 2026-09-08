# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prospective Bucket S selection from the exact full Qwen rating receipt."""

import hashlib
from collections import defaultdict
from math import comb

from experiments.post_training.math_eval.audit_overlay import validated_statuses
from experiments.post_training.math_eval.pool import canonical_json
from experiments.post_training.math_eval.rate import attach_ratings
from experiments.post_training.math_eval.rating_shards import merge_rating_shards


def _sha(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def completed_group_statistics(passes):
    """Report empirical K8 and exchangeable K4-subset statistics, without a CI."""
    if any(type(k) is not int or not 0 <= k <= 8 for k in passes):
        raise ValueError("Expected integer completed counts out of eight")
    if not passes:
        return {"questions": 0, "meets_thresholds": False}
    total = sum(passes)
    informative_numerator = sum(70 - comb(k, 4) - comb(8 - k, 4) for k in passes)
    n = len(passes)
    return {
        "questions": n,
        "completed_pass1": total / (8 * n),
        "k4_subset_informative": informative_numerator / (70 * n),
        "observed_k8_mixed_fraction": sum(0 < k < 8 for k in passes) / n,
        "meets_thresholds": 3 * 8 * n <= 10 * total <= 6 * 8 * n and 5 * informative_numerator > 2 * 70 * n,
    }


def select_qwen_bucket_s(manifest, selection, mechanical_overlay, ratings, generation_audit, *, expected_ratings_sha256):
    """Bind full-population and eligible-row filters before any training adoption.

    Every accepted train question must occur exactly once in the audited K8
    overlay. Full source bins pass the registered numeric thresholds, then only
    measured 0<k<8 rows enter the candidate view. The actual combined view must
    also pass; an unadoptable receipt still reports every bin and exclusion.
    """
    statuses, overlay_sha = validated_statuses(manifest, selection, mechanical_overlay)
    annotated = attach_ratings(manifest, ratings)
    metadata = ratings["metadata"]
    if any(
        metadata.get(key) != value
        for key, value in {
            "model": "qwen",
            "samples": 8,
            "metric": "score_contract_completed",
            "audit_overlay_sha256": overlay_sha,
        }.items()
    ):
        raise ValueError("Bucket S requires the audited Qwen K8 completed-correctness protocol")
    expected = sorted(
        row["prompt_sha256"]
        for row in manifest
        if row["split"] == "train" and statuses[row["prompt_sha256"]] == "accept"
    )
    merge_rating_shards(
        [(ratings, generation_audit)], expected_ids=expected, expected_shard_sha256=[expected_ratings_sha256]
    )
    bins = defaultdict(list)
    membership = set(expected)
    for row in annotated:
        if row["prompt_sha256"] in membership:
            bins[row["bin"]].append(row)
    per_bin, chosen = {}, []
    for name, rows in sorted(bins.items()):
        counts = [row["rating"]["passes"] for row in rows]
        full = completed_group_statistics(counts)
        eligible = [row for row in rows if 0 < row["rating"]["passes"] < 8]
        per_bin[name] = {
            "full_source": full,
            "eligible_subset": completed_group_statistics([row["rating"]["passes"] for row in eligible]),
            "source_selected": full["meets_thresholds"],
            "excluded_zero_of_eight": counts.count(0),
            "excluded_eight_of_eight": counts.count(8),
        }
        if full["meets_thresholds"]:
            chosen.extend(eligible)
    ids = sorted(row["prompt_sha256"] for row in chosen)
    combined = completed_group_statistics([row["rating"]["passes"] for row in chosen])
    result = {
        "schema": "math_eval_qwen_bucket_s_selection_v1",
        "ratings_sha256": expected_ratings_sha256,
        "generation_audit_sha256": _sha(generation_audit),
        "manifest_sha256": metadata["manifest_sha256"],
        "audit_overlay_sha256": overlay_sha,
        "all_source_bins": per_bin,
        "selected_source_bins": [name for name, row in per_bin.items() if row["source_selected"]],
        "prompt_sha256": ids,
        "selected_ids_sha256": _sha(ids),
        "combined": combined,
        "adoptable": combined["meets_thresholds"],
        "scope": (
            "same-data empirical selection; K4 subset exchangeability assumption, no independent latent-probability guarantee"
        ),
    }
    return result | {"selection_sha256": _sha(result)}
