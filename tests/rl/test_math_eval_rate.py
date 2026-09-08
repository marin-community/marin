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
    kwargs = dict(
        expected_ids=["a", "b"],
        samples=2,
        model="qwen",
        checkpoint="initial",
        generation_seed=17,
        temperature=1.0,
        max_response_tokens=2048,
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
    assert rate_from_dump(str(tmp_path), **kwargs)["ratings"][0]["pass_rate_k"] == 0.5
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="audited receipt"):
        rate_from_dump(str(tmp_path), **kwargs)
