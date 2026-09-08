# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.post_training import async_rl_audit as audit


def fixture(tmp_path, provenance):
    dump = tmp_path / "dumped_evals/global_step_0_evals"
    dump.mkdir(parents=True)
    metrics = {
        "eval/all/avg_score": 0.5,
        "eval/all/pass_at_1": 0.5,
        "eval/bin/avg_score": 0.5,
        "eval/bin/pass_at_1": 0.5,
    }
    rows = [
        {
            "uid": str(i),
            "row_ordinal": i,
            "token_provenance": provenance,
            "generator_engine_index": None,
            "prompt_token_ids": [i, 3],
            "response_ids": [4, i],
            "prompt_token_ids_sha256": audit.canonical_sha([i, 3]),
            "response_ids_sha256": audit.canonical_sha([4, i]),
            "response_length": 2,
            "score": [0, i],
            "stop_reason": "stop",
            "data_source": "bin",
        }
        for i in range(2)
    ]
    (dump / "bin.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (dump / "aggregated_results.jsonl").write_text(json.dumps(metrics))
    return metrics


def test_legacy_default_receipt_unchanged(tmp_path):
    metrics = fixture(tmp_path, "finalized_trajectory")
    proof, rows = audit.audit_eval_dump(str(tmp_path), 0, 2, 1, metrics, True)
    assert len(rows) == 2 and proof["aggregate_verified"]
    assert "metric_reference" not in proof
    assert proof["response_metric_scope"].startswith("finalized trajectory")
    assert (
        proof["supplemental_metric_verification"]
        == "each advertised dump/W&B supplemental metric checked; absent legacy metrics allowed"
    )


def test_raw_engine_requires_explicit_provenance_and_scoring_reference(tmp_path):
    metrics = fixture(tmp_path, "raw_engine_response")
    with pytest.raises(ValueError, match="Unexpected token provenance"):
        audit.audit_eval_dump(str(tmp_path), 0, 2, 1, metrics, True)
    with pytest.raises(ValueError, match="Unsupported evaluation token/metric provenance pair"):
        audit.audit_eval_dump(str(tmp_path), 0, 2, 1, metrics, True, expected_token_provenance="raw_engine_response")
    proof, rows = audit.audit_eval_dump(
        str(tmp_path),
        0,
        2,
        1,
        metrics,
        True,
        expected_token_provenance="raw_engine_response",
        metric_reference="serving_score_receipt",
    )
    assert len(rows) == 2 and proof["aggregate_verified"]
    assert proof["metric_reference"] == "serving_score_receipt"
    assert proof["token_provenance"] == "raw_engine_response"
    assert proof["response_metric_scope"].startswith("raw engine")
    assert "W&B" not in proof["supplemental_metric_verification"]
    metrics["eval/all/avg_score"] = 0
    with pytest.raises(ValueError, match="serving score receipt differs"):
        audit.audit_eval_dump(
            str(tmp_path),
            0,
            2,
            1,
            metrics,
            True,
            expected_token_provenance="raw_engine_response",
            metric_reference="serving_score_receipt",
        )
