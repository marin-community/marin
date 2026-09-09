# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json

import pytest

from experiments.post_training.async_rl_m2_diagnostic import audit_diagnostic_source, diagnostic_prerequisites


def prerequisites():
    metrics = {"raw_grad_norm": 1.0, "m2_mask/m2_before": 0.05, "m2_mask/m2_after": 0.02, "m2_mask/masked_fraction": 0.4}
    numerical = {
        "status": "PASS",
        "source_pair": {"marinskyrl": "runtime"},
        "numerical": {
            "loss_absolute_error_max": 0.0,
            "raw_grad_norm_absolute_error_max": 0.0,
            "updated_logprob_absolute_error_max": 0.0,
        },
        "worker_statuses": {"groups": [[{"metrics": copy.deepcopy(metrics)} for _ in range(4)] for _ in range(2)]},
    }
    n1 = {"clean_end_to_end": True, "errors": [], "correction_reports": {}, "vectors": {}}
    for label in ("bc17", "bc29", "mask17", "m2_n1"):
        n1["correction_reports"][label] = {
            "status": "CORRECTION_HISTORY_PASS",
            "successful_updates": 8,
            "aggregate_field_checks": 112,
            "update_field_checks": 112,
            "minimum_raw_gradient_norm": 1.0,
        }
        n1["vectors"][label] = {"8": [[str(i), f"hash{i}", 1, "stop"] for i in range(128)]}
    return numerical, n1


def test_diagnostic_prerequisites_leave_missing_native_comparison_unqualified():
    numerical, n1 = prerequisites()
    result = diagnostic_prerequisites(numerical, n1, runtime_sha="runtime")
    assert result["status"] == "M2_N8_DIAGNOSTIC_PREREQUISITES_PASS"
    assert result["matched_n1_uid_union_qualified"] is False and result["full_k12_qualified"] is False
    numerical["worker_statuses"]["groups"][1][0]["metrics"]["raw_grad_norm"] = 0
    with pytest.raises(ValueError, match="vacuous"):
        diagnostic_prerequisites(numerical, n1, runtime_sha="runtime")


def test_diagnostic_rejects_failed_numerics_or_fresh_data_noise():
    numerical, n1 = prerequisites()
    numerical["numerical"]["updated_logprob_absolute_error_max"] = 0.01
    with pytest.raises(ValueError, match="parity"):
        diagnostic_prerequisites(numerical, n1, runtime_sha="runtime")
    numerical, n1 = prerequisites()
    n1["vectors"]["m2_n1"]["8"][0][2] = 0
    with pytest.raises(ValueError, match="noise-band"):
        diagnostic_prerequisites(numerical, n1, runtime_sha="runtime")


def test_diagnostic_source_requires_all_512_actual_unique_groups():
    chunks = {offset: [f"uid{i}" for i in range(offset, offset + 64)] for offset in range(0, 512, 64)}
    uids = [uid for chunk in chunks.values() for uid in chunk]
    digest = int(hashlib.sha256(json.dumps(sorted(set(uids))).encode()).hexdigest()[:13], 16)
    assert audit_diagnostic_source(chunks, wandb_digest=digest)["source_groups"] == 512
    incomplete = {key: value for key, value in chunks.items() if key != 448}
    with pytest.raises(ValueError, match="offsets"):
        audit_diagnostic_source(incomplete, wandb_digest=digest)
    chunks[448][0] = chunks[0][0]
    with pytest.raises(ValueError, match="distinct"):
        audit_diagnostic_source(chunks, wandb_digest=digest)
