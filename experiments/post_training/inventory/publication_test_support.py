# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json

from experiments.post_training.inventory import publish_loop_result as publication
from infra.marina.apps.rl_progress.server.app import comparison_fields


def result_record():
    return {
        "run_id": "test/control",
        "attempt_id": "attempt-1",
        "label": "control",
        "run_date": "2026-11-02T00:00:00Z",
        "date_basis": "submission_acknowledged",
        "started_at": "2026-11-02T00:00:00Z",
        "completed_at": "2026-11-02T01:00:00Z",
        "elapsed_seconds": 3600,
        "timing_scope": "coordinator_submission_to_terminal",
        "status": "succeeded",
        "optimizer_steps": 25,
        "source_revision": "source-1",
        "data_identity": "data-1",
        "initializer_identity": "model-1",
        "evaluation": {
            "name": "fast",
            "version": "1",
            "task_count": 100,
            "task_set_sha256": "a" * 64,
            "protocol_sha256": None,
        },
        "baseline": {"reward": 0.2, "attempted": 100, "scored": 100},
        "final": {"reward": 0.4, "attempted": 100, "scored": 100},
        "artifact_uri": "s3://example/test",
        "evidence": ["synthetic unit-test result"],
    }


class ReportTransport:
    def __init__(self):
        self.rows = {}
        self.fail_readback = False

    def __call__(self, origin, method, path, *, json_body=None):
        if path.startswith("/api/marina/applets/"):
            return {"authenticated_url": f"https://applets.marina.oa.dev/a/{publication.APPLET_ID}/"}
        if method == "POST":
            parameters = json_body["parameters"]
            key = (parameters["run_id"], parameters["attempt_id"])
            if "record" in parameters:
                if key in self.rows:
                    return {"row_count": 0}
                self.rows[key] = json.loads(parameters["record"])
                return {"row_count": 1}
            if self.fail_readback:
                self.fail_readback = False
                raise ConnectionError("synthetic outage after insertion")
            return {"rows": [{"record": copy.deepcopy(self.rows[key])}]}
        return {"runs": [record | comparison_fields(record) for record in self.rows.values()]}
