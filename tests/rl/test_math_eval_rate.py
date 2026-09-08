# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.math_eval.pool import canonical_json
from experiments.post_training.math_eval.rate import attach_ratings, rate_from_dump, rate_from_records


def fixture():
    manifest = [{"prompt_sha256": "a", "split": "train"}, {"prompt_sha256": "b", "split": "heldout"}]
    records = [
        {
            "prompt_sha256": q,
            "model": "qwen",
            "prompt_template_id": "template",
            "row_ordinal": i,
            "response_tokens": 100,
            "score_contract_completed": int(i == 1),
            "contract_correct": True,
            "score_contract": 1.0,
            "truncated": i != 1,
        }
        for i, q in enumerate(["a", "a", "b", "b"])
    ]
    receipt = {
        "scope": "frozen_pool",
        "contract_metric_parity_verified": True,
        "records": 4,
        "expected_ids_sha256": hashlib.sha256(canonical_json(["a", "b"]).encode()).hexdigest(),
        "manifest_sha256": hashlib.sha256(canonical_json(manifest).encode()).hexdigest(),
        "records_sha256": "e" * 64,
        "audit_overlay_sha256": "d" * 64,
        "tokenizer_sha256": "c" * 64,
        "prompt_template_id": "template",
    }
    receipt["audit"] = {
        "step": 0,
        "dump_namespace": None,
        "rows": 4,
        "unique_uids": 2,
        "present": True,
        "aggregate_verified": True,
        "ordered_prompt_sha256": "1" * 64,
        "ordered_response_sha256": "2" * 64,
        "ordered_result_sha256": "3" * 64,
    }
    generation_audit = {
        "training_evidence_pass": True,
        "clean_end_to_end": True,
        "run_id": "fixture",
        "attempt_id": "attempt",
        "storage": {"request_fingerprint": "4" * 64},
        "resolved_config": {
            "trainer.seed": 17,
            "trainer.eval_before_train": True,
            "generator.eval_sampling_params": {"temperature": 1.0, "max_generate_length": 2048},
        },
        "provenance": {"seed": 17, "model": {"identity": "initial"}},
        "eval_dumps": [deepcopy(receipt["audit"])],
    }
    kwargs = dict(
        expected_ids=["a", "b"],
        samples=2,
        model="qwen",
        checkpoint="initial",
        engine_global_seed=17,
        temperature=1.0,
        max_response_tokens=2048,
        generation_audit=generation_audit,
        generation_audit_sha256=hashlib.sha256(canonical_json(generation_audit).encode()).hexdigest(),
    )
    return manifest, records, receipt, kwargs


def test_ratings_use_completed_correctness_and_leave_frozen_manifest_unchanged():
    manifest, records, receipt, kwargs = fixture()
    before = deepcopy(manifest)
    result = rate_from_records(records, receipt, **kwargs)
    assert [row["pass_rate_k"] for row in result["ratings"]] == [0.5, 0]
    assert [row["contract_correct_rate"] for row in result["ratings"]] == [1, 1]
    joined = attach_ratings(manifest, result)
    assert joined[0]["rating"]["pass_rate_k"] == 0.5
    assert manifest == before and [row["split"] for row in joined] == ["train", "heldout"]


@pytest.mark.parametrize("alteration", ["missing", "duplicate", "foreign", "model", "k"])
def test_ratings_refuse_incomplete_or_mixed_sampling_protocol(alteration):
    _manifest, records, receipt, kwargs = fixture()
    if alteration == "missing":
        records.pop()
    elif alteration == "duplicate":
        records[0]["row_ordinal"] = records[1]["row_ordinal"]
    elif alteration == "foreign":
        records[0]["prompt_sha256"] = "foreign"
    elif alteration == "model":
        records[0]["model"] = "snowball"
    else:
        kwargs["samples"] = 4
    with pytest.raises(ValueError):
        rate_from_records(records, receipt, **kwargs)


def test_rating_parquet_roundtrip_checks_exact_audited_bytes(tmp_path):
    _manifest, records, receipt, kwargs = fixture()
    path = tmp_path / "records.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)
    receipt["records_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "summary.json").write_text(json.dumps(receipt))
    audit = kwargs.pop("generation_audit")
    audit_path = tmp_path / "generation-audit.json"
    audit_path.write_text(json.dumps(audit))
    kwargs["generation_audit_uri"] = str(audit_path)
    assert rate_from_dump(str(tmp_path), **kwargs)["ratings"][0]["pass_rate_k"] == 0.5
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="audited receipt"):
        rate_from_dump(str(tmp_path), **kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("engine_global_seed", 18),
        ("temperature", 0.5),
        ("checkpoint", "other-checkpoint"),
        ("max_response_tokens", 4096),
    ],
)
def test_claimed_generation_values_cannot_override_audited_configuration(field, value):
    _manifest, records, receipt, kwargs = fixture()
    kwargs[field] = value
    with pytest.raises(ValueError):
        rate_from_records(records, receipt, **kwargs)


@pytest.mark.parametrize("alteration", ["digest", "response", "phase", "failed"])
def test_generation_audit_must_match_exact_response_receipt(alteration):
    _manifest, records, receipt, kwargs = fixture()
    audit = kwargs["generation_audit"]
    if alteration == "response":
        audit["eval_dumps"][0]["ordered_response_sha256"] = "f" * 64
    elif alteration == "phase":
        audit["eval_dumps"][0]["step"] = receipt["audit"]["step"] = 50
    elif alteration == "failed":
        audit["clean_end_to_end"] = False
    if alteration != "digest":
        kwargs["generation_audit_sha256"] = hashlib.sha256(canonical_json(audit).encode()).hexdigest()
    else:
        kwargs["generation_audit_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        rate_from_records(records, receipt, **kwargs)


def test_trusted_reduction_cannot_certify_ratings_for_selection():
    manifest, records, receipt, kwargs = fixture()
    kwargs.pop("generation_audit")
    kwargs.pop("generation_audit_sha256")
    result = rate_from_records(records, receipt, **kwargs)
    assert result["metadata"]["generation_protocol_verified"] is False
    with pytest.raises(ValueError, match="Unverified"):
        attach_ratings(manifest, result)


def test_temperature_one_claim_cannot_certify_a_greedy_generation_receipt():
    _manifest, records, receipt, kwargs = fixture()
    audit = kwargs["generation_audit"]
    audit["resolved_config"]["generator.eval_sampling_params"]["temperature"] = 0.0
    kwargs["generation_audit_sha256"] = hashlib.sha256(canonical_json(audit).encode()).hexdigest()
    with pytest.raises(ValueError, match="Claimed sampling protocol"):
        rate_from_records(records, receipt, **kwargs)
