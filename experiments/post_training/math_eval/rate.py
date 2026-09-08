# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Difficulty ratings from proven response tables, stored separately from frozen membership."""

import hashlib
import json
import re
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.math_eval.contract import QWEN, SNOWBALL
from experiments.post_training.math_eval.pool import canonical_json
from experiments.post_training.math_eval.rendering import INCREMENTAL_RENDERING

# Startup-only profiles qualified in contract-v1.md; changing model content,
# tokenizer, template or renderer requires another reviewed profile.
MODEL_PROFILES = {
    "qwen": {
        "identity_pattern": r"users/ahmad/models/async-rl-qwen3-0\.6b@[^:]+:8a30d2b5",
        "tokenizer_uri": "Qwen/Qwen3-0.6B",
        "tokenizer_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "tokenizer_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "prompt_template_id": QWEN.template_id,
        "renderer_qualification_sha256": "7992b6a8defc71150a6efaf617122bd5fb6cb57f528b253893d4136a7724d545",
        "native_runtime_commit": "d840cd29665a12c9459911b2f8cbc2245f176d6f",
        "max_prompt_tokens": 1024,
        "max_response_tokens": 2048,
    },
    "snowball": {
        "identity_pattern": r"models/snowball-67b-a2b-sft-s2-thinking@2026\.08\.30:c6168770",
        "tokenizer_uri": "marin-community/marin-tokenizer",
        "tokenizer_revision": "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2",
        "tokenizer_sha256": "881c9c36c359e1617afef6f7583403567931b7b4f43f6552d2b2155a131650a2",
        "prompt_template_id": SNOWBALL.template_id,
        "renderer_qualification_sha256": "2a4b5edb5ca3f1c0a2b1a9e5370576bc344b56e1989e5225b4bfffdf9454551c",
        "native_runtime_commit": "d840cd29665a12c9459911b2f8cbc2245f176d6f",
        "max_prompt_tokens": 4096,
        "max_response_tokens": 8192,
    },
}


def empirical_difficulty(passes, samples):
    """Label measured completed correctness; this is not a latent probability estimate."""
    if type(passes) is not int or type(samples) is not int or samples <= 0 or not 0 <= passes <= samples:
        raise ValueError("Invalid completed-correctness count")
    rate = passes / samples
    if rate == 0:
        return "beyond"
    for upper, label in ((0.1, "frontier"), (0.3, "hard"), (0.6, "mid"), (0.9, "easy")):
        if rate <= upper:
            return label
    return "mastered"


def rate_from_dump(harness_output_uri, *, generation_audit_uri, generation_audit_sha256, **kwargs):
    """Read a proven record parquet, checking its exact receipt hash before reducing."""
    receipt = json.loads(StoragePath(harness_output_uri + "/summary.json").read_bytes())
    content = StoragePath(harness_output_uri + "/records.parquet").read_bytes()
    if hashlib.sha256(content).hexdigest() != receipt["records_sha256"]:
        raise ValueError("Rating record parquet differs from its audited receipt")
    records = pq.read_table(pa.BufferReader(content)).to_pylist()
    generation_audit = json.loads(StoragePath(generation_audit_uri).read_bytes())
    return rate_from_records(
        records, receipt, generation_audit=generation_audit, generation_audit_sha256=generation_audit_sha256, **kwargs
    )


def rate_from_records(
    records,
    receipt,
    *,
    expected_ids,
    samples,
    model,
    checkpoint,
    engine_global_seed,
    temperature,
    max_response_tokens,
    generation_audit=None,
    generation_audit_sha256=None,
):
    """Rate every sampled question from exactly K completed-correctness observations.

    The harness receipt must certify frozen-pool identity and aggregate parity. Raw
    optimization rewards and exact correctness remain separate diagnostic means.
    Returning a ratings overlay leaves the candidate pool's immutable hash intact.
    Without a hash-pinned generation audit this is only a trusted-input reduction;
    its unverified overlay cannot be attached to a pool for selection.
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
    protocol = _generation_protocol(receipt, generation_audit, generation_audit_sha256)
    if protocol is not None and any(
        protocol[key] != value
        for key, value in {
            "checkpoint": checkpoint,
            "model_label": model,
            "engine_global_seed": engine_global_seed,
            "temperature": temperature,
            "max_response_tokens": max_response_tokens,
            "samples": samples,
        }.items()
    ):
        raise ValueError("Claimed sampling protocol differs from the audited generation configuration")
    if protocol is not None:
        rendering = protocol.get("contract_response_rendering", "decode_skip_special_tokens")
        if rendering not in {"decode_skip_special_tokens", INCREMENTAL_RENDERING} or any(
            row.get("contract_response_rendering") != rendering for row in records
        ):
            raise ValueError("Rating rows lack qualified native response rendering")
    metadata = {
        "model": model,
        "checkpoint": checkpoint,
        "engine_global_seed": engine_global_seed,
        "generation_protocol_verified": protocol is not None,
        "generation_audit_sha256": generation_audit_sha256,
        "generation_provenance": protocol,
        "generated_at_utc": None if protocol is None else protocol.get("generated_at_utc"),
        "dump_uri": None if protocol is None else protocol.get("dump_uri"),
        "difficulty_category_basis": "empirical_completed_correctness",
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
            "passes": sum(int(row["score_contract_completed"]) for row in rows),
            "mean_response_tokens": sum(row["response_tokens"] for row in rows) / samples,
            "empirical_difficulty": empirical_difficulty(
                sum(int(row["score_contract_completed"]) for row in rows), samples
            ),
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
    if not ratings_overlay["metadata"].get("generation_protocol_verified"):
        raise ValueError("Unverified generation protocol cannot select pool membership")
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


def _generation_protocol(receipt, generation_audit, expected_sha256):
    """Bind an existing terminal audit to these exact responses and its startup model.

    The accepted digest is the canonical JSON SHA of a single audit_run result.
    Native training dumps support startup checkpoints; inference-only dumps use
    a separate terminal and raw-token serving adapter. The engine/global
    seed initializes engines, and is not claimed as a per-request sampling seed.
    """
    if generation_audit is None:
        if expected_sha256 is not None:
            raise ValueError("Generation audit is missing")
        return None
    if hashlib.sha256(canonical_json(generation_audit).encode()).hexdigest() != expected_sha256:
        raise ValueError("Generation audit differs from its immutable digest")
    if generation_audit.get("schema") == "math_eval_serving_audit_v1":
        from experiments.post_training.math_eval.serving_audit import protocol_from_serving_audit  # noqa: PLC0415

        return protocol_from_serving_audit(receipt, generation_audit)
    if not generation_audit.get("training_evidence_pass") or not generation_audit.get("clean_end_to_end"):
        raise ValueError("Generation requires a successful terminal audit")
    endpoint = receipt["audit"]
    keys = (
        "step",
        "dump_namespace",
        "rows",
        "unique_uids",
        "ordered_prompt_sha256",
        "ordered_response_sha256",
        "ordered_result_sha256",
    )
    matching = [dump for dump in generation_audit["eval_dumps"] if all(dump[key] == endpoint[key] for key in keys)]
    if len(matching) != 1 or not endpoint["present"] or not endpoint["aggregate_verified"]:
        raise ValueError("Generation audit does not certify these exact response records")
    if endpoint["step"] != 0:
        raise ValueError("Only startup model checkpoint provenance is supported")
    cfg = generation_audit["resolved_config"]
    sampling = cfg["generator.eval_sampling_params"]
    if not cfg["trainer.eval_before_train"] or cfg["trainer.seed"] != generation_audit["provenance"]["seed"]:
        raise ValueError("Startup generation seed or checkpoint phase differs")
    rows, questions = endpoint["rows"], endpoint["unique_uids"]
    if questions <= 0 or rows % questions:
        raise ValueError("Audited per-question sample count is invalid")
    audited_model = generation_audit["provenance"]["model"]
    matches = [
        (label, profile)
        for label, profile in MODEL_PROFILES.items()
        if re.fullmatch(profile["identity_pattern"], audited_model["identity"])
    ]
    if len(matches) != 1:
        raise ValueError("Audited model identity has no qualified rating profile")
    label, profile = matches[0]
    if (
        any(audited_model.get(key) != profile[key] for key in ("tokenizer_uri", "tokenizer_revision"))
        or generation_audit["storage"].get("runtime", {}).get("commit") != profile["native_runtime_commit"]
        or any(receipt.get(key) != profile[key] for key in ("tokenizer_sha256", "prompt_template_id"))
        or sampling["max_generate_length"] != profile["max_response_tokens"]
        or cfg.get("generator.max_input_length") != profile["max_prompt_tokens"]
        or sampling.get("top_p") != 1.0
    ):
        raise ValueError("Rating model/tokenizer/template or token protocol differs from its qualified profile")
    return {
        "model_label": label,
        "qualified_profile_sha256": hashlib.sha256(canonical_json(profile).encode()).hexdigest(),
        "renderer_qualification_sha256": profile["renderer_qualification_sha256"],
        "tokenizer_sha256": profile["tokenizer_sha256"],
        "max_prompt_tokens": profile["max_prompt_tokens"],
        "top_p": sampling["top_p"],
        "checkpoint": generation_audit["provenance"]["model"]["identity"],
        "model": generation_audit["provenance"]["model"],
        "engine_global_seed": cfg["trainer.seed"],
        "request_sampling_seed": sampling.get("seed"),
        "temperature": sampling["temperature"],
        "max_response_tokens": sampling["max_generate_length"],
        "samples": rows // questions,
        "run_id": generation_audit["run_id"],
        "attempt_id": generation_audit["attempt_id"],
        "request_fingerprint": generation_audit["storage"]["request_fingerprint"],
        "step": 0,
    }
