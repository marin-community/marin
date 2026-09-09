# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evidence contracts for a standalone within-batch M2 diagnostic."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

from experiments.post_training.async_rl_correction_audit import audit_completed_noise_band


def diagnostic_prerequisites(numerical: Mapping, n1: Mapping, *, runtime_sha: str) -> dict:
    """Read numerical and fresh-data evidence without asserting the missing N1 native gate."""
    if numerical["status"] != "PASS" or numerical["source_pair"]["marinskyrl"] != runtime_sha:
        raise ValueError("Numerical source or status differs")
    result = numerical["numerical"]
    for key, tolerance in (
        ("loss_absolute_error_max", 1e-6),
        ("raw_grad_norm_absolute_error_max", 1e-6),
        ("updated_logprob_absolute_error_max", 1e-4),
    ):
        value = result[key]
        if not math.isfinite(value) or not 0 <= value <= tolerance:
            raise ValueError("Numerical parity failed")
    groups = numerical["worker_statuses"]["groups"]
    if len(groups) != 2 or any(len(group) != 4 for group in groups):
        raise ValueError("Numerical worker coverage is incomplete")
    for group_index, group in enumerate(groups):
        for worker in group:
            metrics = worker["metrics"]
            if not math.isfinite(metrics["raw_grad_norm"]) or metrics["raw_grad_norm"] <= 0:
                raise ValueError("Numerical M2 fixture is vacuous")
            if group_index == 0:
                if not metrics["m2_mask/m2_before"] > 0.04:
                    raise ValueError("Numerical M2 fixture never crossed the threshold")
                if not (0 < metrics["m2_mask/masked_fraction"] < 1 and metrics["m2_mask/m2_after"] < 0.04):
                    raise ValueError("Numerical M2 fixture lacks retained support")
    if not n1["clean_end_to_end"] or n1["errors"]:
        raise ValueError("Fresh-data durable audit failed")
    labels = {"bc17", "bc29", "mask17", "m2_n1"}
    if set(n1["correction_reports"]) != labels:
        raise ValueError("Fresh-data arm coverage differs")
    for report in n1["correction_reports"].values():
        if report["status"] != "CORRECTION_HISTORY_PASS" or report["successful_updates"] != 8:
            raise ValueError("Fresh-data history did not pass")
        if report["aggregate_field_checks"] != 112 or report["update_field_checks"] != 112:
            raise ValueError("Fresh-data correction schema is incomplete")
        if not math.isfinite(report["minimum_raw_gradient_norm"]) or report["minimum_raw_gradient_norm"] <= 0:
            raise ValueError("Fresh-data gradients are vacuous")
    vectors = {}
    for label in labels:
        rows = n1["vectors"][label]["8"]
        vectors[label] = {row[1]: int(row[2] == 1 and row[3] in {"complete", "end_turn", "eos", "stop"}) for row in rows}
        if len(vectors[label]) != 128 or len(rows) != 128:
            raise ValueError("Fresh-data final evaluation coverage differs")
    ids = sorted(vectors["bc17"])
    if any(set(values) != set(ids) for values in vectors.values()):
        raise ValueError("Fresh-data evaluation hashes differ")
    scores = {label: [values[uid] for uid in ids] for label, values in vectors.items()}
    bands = {
        label: audit_completed_noise_band(scores["bc17"], scores["bc29"], scores[label]) for label in ("mask17", "m2_n1")
    }
    if not all(result["within_noise_band"] for result in bands.values()):
        raise ValueError("Fresh-data engineering noise-band check failed")
    return {
        "status": "M2_N8_DIAGNOSTIC_PREREQUISITES_PASS",
        "runtime_sha": runtime_sha,
        "numerical_workers": 8,
        "fresh_data_updates": {label: 8 for label in sorted(labels)},
        "noise_bands": bands,
        "n1_native_coverage": "incomplete; not required by this standalone mechanism diagnostic",
        "matched_n1_uid_union_qualified": False,
        "full_k12_qualified": False,
    }


def audit_diagnostic_source(chunks: Mapping[int, Sequence[str]], *, wandb_digest: int) -> dict:
    """Verify this run's full source vector; do not infer the missing N1 vector union."""
    if set(chunks) != set(range(0, 512, 64)):
        raise ValueError("Missing or duplicate diagnostic source offsets")
    if any(len(values) != 64 for values in chunks.values()):
        raise ValueError("Incomplete diagnostic source chunk")
    uids = [uid for offset in sorted(chunks) for uid in chunks[offset]]
    if any(not isinstance(uid, str) or not uid for uid in uids) or len(set(uids)) != 512:
        raise ValueError("The diagnostic requires 512 distinct source groups")
    digest = int(hashlib.sha256(json.dumps(sorted(set(uids))).encode()).hexdigest()[:13], 16)
    if type(wandb_digest) is not int or digest != wandb_digest:
        raise ValueError("Source vector disagrees with the exact integer W&B digest")
    return {"source_groups": 512, "digest_matches": True, "matched_n1_uid_union_qualified": False}
