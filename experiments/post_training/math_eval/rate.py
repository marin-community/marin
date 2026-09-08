# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Difficulty ratings from proven response tables, stored separately from frozen membership."""

import hashlib
import json
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.math_eval.pool import canonical_json


def rate_from_dump(harness_output_uri, **kwargs):
    """Read a proven record parquet, checking its exact receipt hash before reducing."""
    receipt = json.loads(StoragePath(harness_output_uri + "/summary.json").read_bytes())
    content = StoragePath(harness_output_uri + "/records.parquet").read_bytes()
    if hashlib.sha256(content).hexdigest() != receipt["records_sha256"]:
        raise ValueError("Rating record parquet differs from its audited receipt")
    records = pq.read_table(pa.BufferReader(content)).to_pylist()
    return rate_from_records(records, receipt, **kwargs)


def rate_from_records(
    records, receipt, *, expected_ids, samples, model, checkpoint, generation_seed, temperature, max_response_tokens
):
    """Rate every sampled question from exactly K completed-correctness observations.

    The harness receipt must certify frozen-pool identity and aggregate parity. Raw
    optimization rewards and exact correctness remain separate diagnostic means.
    Returning a ratings overlay leaves the candidate pool's immutable hash intact.
    """
    if (
        receipt.get("scope") != "frozen_pool"
        or not receipt.get("contract_metric_parity_verified")
        or receipt.get("records") != len(records)
        or samples <= 0
        or temperature != 1.0
        or max_response_tokens <= 0
    ):
        raise ValueError("Ratings require proven frozen records and the declared K/temperature/token protocol")
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Ratings require unique frozen question membership")
    expected_hash = hashlib.sha256(canonical_json(sorted(expected_ids)).encode()).hexdigest()
    if receipt["expected_ids_sha256"] != expected_hash:
        raise ValueError("Ratings receipt certifies different expected questions")
    grouped = defaultdict(list)
    ordinals = set()
    for row in records:
        if (
            row["model"] != model
            or row["prompt_template_id"] != receipt["prompt_template_id"]
            or row["row_ordinal"] in ordinals
            or row["response_tokens"] > max_response_tokens
            or row["score_contract_completed"] not in (0, 1)
        ):
            raise ValueError("Rating response metadata, sample identity or completion score changed")
        ordinals.add(row["row_ordinal"])
        grouped[row["prompt_sha256"]].append(row)
    if set(grouped) != set(expected_ids) or any(len(rows) != samples for rows in grouped.values()):
        raise ValueError("Ratings question membership or per-question K differs")
    metadata = {
        "model": model,
        "checkpoint": checkpoint,
        "generation_seed": generation_seed,
        "temperature": temperature,
        "max_response_tokens": max_response_tokens,
        "samples": samples,
        "manifest_sha256": receipt["manifest_sha256"],
        "records_sha256": receipt["records_sha256"],
        "prompt_template_id": receipt["prompt_template_id"],
        "tokenizer_sha256": receipt["tokenizer_sha256"],
        "audit_overlay_sha256": receipt["audit_overlay_sha256"],
        "metric": "score_contract_completed",
    }
    ratings = [
        {
            "prompt_sha256": digest,
            "samples": samples,
            "pass_rate_k": sum(row["score_contract_completed"] for row in rows) / samples,
            "contract_correct_rate": sum(row["contract_correct"] for row in rows) / samples,
            "native_score_mean": sum(row["score_contract"] for row in rows) / samples,
            "truncated_fraction": sum(row["truncated"] for row in rows) / samples,
        }
        for digest, rows in sorted(grouped.items())
    ]
    result = {"metadata": metadata, "ratings": ratings}
    return result | {"ratings_sha256": hashlib.sha256(canonical_json(result).encode()).hexdigest()}


def attach_ratings(manifest, ratings_overlay):
    """Return a derived join view; never alter frozen source rows or split assignments."""
    manifest_hash = hashlib.sha256(
        canonical_json(sorted(manifest, key=lambda row: row["prompt_sha256"])).encode()
    ).hexdigest()
    if manifest_hash != ratings_overlay["metadata"]["manifest_sha256"]:
        raise ValueError("Ratings belong to another candidate manifest")
    body = {key: value for key, value in ratings_overlay.items() if key != "ratings_sha256"}
    if hashlib.sha256(canonical_json(body).encode()).hexdigest() != ratings_overlay["ratings_sha256"]:
        raise ValueError("Ratings overlay content changed")
    lookup = {row["prompt_sha256"]: row for row in ratings_overlay["ratings"]}
    if len(lookup) != len(ratings_overlay["ratings"]) or not lookup.keys() <= {row["prompt_sha256"] for row in manifest}:
        raise ValueError("Ratings membership is duplicated or foreign")
    return [row | {"rating": lookup.get(row["prompt_sha256"])} for row in manifest]
