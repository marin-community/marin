# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine disjoint question shards while retaining each native generation audit."""

import hashlib

from experiments.post_training.math_eval.pool import canonical_json

METADATA_KEYS = (
    "model",
    "checkpoint",
    "engine_global_seed",
    "temperature",
    "max_response_tokens",
    "samples",
    "manifest_sha256",
    "prompt_template_id",
    "tokenizer_sha256",
    "audit_overlay_sha256",
    "metric",
    "difficulty_category_basis",
)
PROTOCOL_KEYS = (
    "producer_source_commit",
    "model_label",
    "checkpoint",
    "engine_global_seed",
    "request_sampling_seed",
    "temperature",
    "top_p",
    "max_prompt_tokens",
    "max_response_tokens",
    "samples",
    "tokenizer_sha256",
    "prompt_template_id",
    "execution_cluster",
    "gpu_variant",
    "gpu_count",
)
RUNTIME_KEYS = ("source_commit", "version", "api_version", "wheel_sha256", "wheel_url", "compute_capability")


def _sha(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def merge_rating_shards(shards, *, expected_ids, expected_shard_sha256):
    """Join immutable overlay/audit pairs, requiring one K-sample shard per question.

    A merged overlay has no single record parquet or native attempt. Its source
    shards retain those identities; it must not be represented as one generation.
    """
    if not shards or not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Rating merge requires shards and unique expected questions")
    if len(expected_shard_sha256) != len(shards) or len(set(expected_shard_sha256)) != len(shards):
        raise ValueError("Rating merge requires a distinct pinned digest for each input shard")
    rows, sources, seen = [], [], set()
    common = common_protocol = None
    for (overlay, generation_audit), expected_digest in zip(shards, expected_shard_sha256, strict=True):
        metadata = overlay["metadata"]
        protocol = generation_audit["protocol"]
        digest = overlay["ratings_sha256"]
        ids = [row["prompt_sha256"] for row in overlay["ratings"]]
        if (
            digest != expected_digest
            or _sha({key: value for key, value in overlay.items() if key != "ratings_sha256"}) != digest
            or metadata.get("generation_protocol_verified") is not True
            or _sha(generation_audit) != metadata.get("generation_audit_sha256")
            or generation_audit.get("schema") != "math_eval_serving_audit_v1"
            or generation_audit.get("inference_evidence_pass") is not True
            or generation_audit.get("clean_end_to_end") is not True
            or protocol != metadata.get("generation_provenance")
            or generation_audit.get("records_sha256") != metadata.get("records_sha256")
            or generation_audit.get("expected_ids_sha256") != _sha(sorted(ids))
        ):
            raise ValueError("Rating shard differs from its immutable generation audit")
        signature = {key: metadata[key] for key in METADATA_KEYS}
        sampling = {key: protocol[key] for key in PROTOCOL_KEYS}
        sampling["runtime"] = {key: protocol["runtime"][key] for key in RUNTIME_KEYS}
        if common is None:
            common, common_protocol = signature, sampling
        if signature != common or sampling != common_protocol:
            raise ValueError("Rating shards have different model, sampling, verifier or runtime protocols")
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Rating shards contain duplicate question/sample membership")
        if any(
            row["samples"] != metadata["samples"]
            or not 0 <= row["passes"] <= row["samples"]
            or row["pass_rate_k"] != row["passes"] / row["samples"]
            for row in overlay["ratings"]
        ):
            raise ValueError("Rating shard changes its audited per-question sample count or completion rate")
        seen.update(ids)
        rows.extend(row | {"source_shard_sha256": digest} for row in overlay["ratings"])
        sources.append({"ratings_sha256": digest, "metadata": metadata, "questions": len(ids)})
    if seen != set(expected_ids):
        raise ValueError("Rating shards do not cover exactly the frozen expected questions")
    result = {
        "schema": "math_eval_rating_shards_v1",
        "metadata": common | {"generation_protocol_verified": True},
        "source_shards": sorted(sources, key=lambda source: source["ratings_sha256"]),
        "ratings": sorted(rows, key=lambda row: row["prompt_sha256"]),
    }
    return result | {"ratings_sha256": _sha(result)}
