# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A failed local reason audit cannot reach the external job scheduler."""

import json

import pytest

from experiments.post_training.task_curation.difficulty import audit
from experiments.post_training.task_curation.difficulty.legacy import LegacyReasonError


def test_launch_gate_blocks_unknown_reasons_before_any_scheduler_side_effect(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "base": "s3://bucket/",
                "runs_discovered": 1,
                "unattributed_runs": [],
                "solvers": {
                    "small": {
                        "runs": [
                            {
                                "root": "s3://bucket/run",
                                "catalog": {
                                    "pending_reasons": {"awaiting_scored_coverage": 1},
                                    "not_measured_reasons": {},
                                },
                            }
                        ]
                    }
                },
            }
        )
    )
    log = tmp_path / "audit.log"
    log.write_text("Complete reason audit failed:\nUnmapped difficulty gap reason: 'unknown_model_wait'\n")
    submitted = tmp_path / "external-submission.json"

    def scheduler_submit(command, check):
        submitted.write_text(json.dumps({"accepted": True}))

    monkeypatch.setattr(audit.subprocess, "run", scheduler_submit)
    with pytest.raises(LegacyReasonError):
        audit.launch_import(report, log, "s3://bucket/runs", "s3://bucket/ledger", "import-proof", ())
    assert not submitted.exists()
    log.write_text(
        "Complete reason audit failed:\nUnmapped difficulty gap reason: 'Awaiting shared model provider capacity'\n"
    )
    proof = audit.launch_import(report, log, "s3://bucket/runs", "s3://bucket/ledger", "import-proof", ())
    assert proof.run_roots == ("s3://bucket/run",)
    assert json.loads(submitted.read_text()) == {"accepted": True}
