# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.evaluation.recover_sweep_record import audit_resume_rows


def test_selective_retry_preserves_successes_and_rejects_incomplete_or_changed_responses():
    def row(key, output):
        return {
            "namespace": "gsm",
            "unit_key": key,
            "payload": json.dumps({"sample": {"source_id": key, "prompt": "question"}, "output": output}),
        }

    good = row("good", "answer")
    failed = row("retry", {"__evalchemy_resume_value__": "failed_generation", "failure_category": "model_transport"})
    fixed = row("retry", "new answer")
    assert audit_resume_rows([good, failed], [fixed, good]) == {
        "preserved_responses": 1,
        "retried_transport_failures": 1,
    }
    with pytest.raises(ValueError, match="previously successful"):
        audit_resume_rows([good, failed], [row("good", "different"), fixed])
    with pytest.raises(ValueError, match="still contains"):
        audit_resume_rows([good, failed], [good, failed])
    with pytest.raises(ValueError, match="identity set"):
        audit_resume_rows([good, failed], [good])
