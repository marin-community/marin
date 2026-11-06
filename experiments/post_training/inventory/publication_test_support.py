# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pyarrow as pa
import pyarrow.parquet as pq
from taskcompendium.convert.answers import answer_task
from taskcompendium.models import Source
from taskcompendium.pipeline.models import RawRow
from verifyit.spec import ExactSpec

from experiments.post_training.inventory import publish_loop_result as publication
from experiments.post_training.task_curation.inventory import TASK_SPEC_COLUMN
from infra.marina.apps.rl_progress.server.app import comparison_fields


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def answer(task, score, *, low=0, high=1, status="verified", tokens=1, error=None):
    verification = {"status": status, "score": score, "score_min": low, "score_max": high}
    return {"task_id": task, "response_token_ids": [1] * tokens, "exception_type": error, "verification": verification}


def task_parquet(path, identifiers):
    template = answer_task(
        RawRow("template", Source(dataset="fixture", revision="1", row="template", importer_revision="1"), {}),
        prompt="Reply yes.",
        spec=ExactSpec(expected=("yes",)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{TASK_SPEC_COLUMN: template.model_copy(update={"id": item}).model_dump_json()} for item in identifiers]
        ),
        path,
    )


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
        self.requests = []
        self.fail_readback = False

    def __call__(self, origin, method, path, *, json_body=None):
        self.requests.append((origin, method, path))
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
