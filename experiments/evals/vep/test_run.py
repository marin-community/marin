# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json

import pytest
from requests import Session

from experiments.evals.vep.run import run_panel


class RecordingClient:
    max_requests = 1000
    max_seconds = 900
    fingerprint = "fixture"

    def __init__(self):
        self.requests_sent = 0
        self.session = Session()
        self.checks = 0
        self.fail_final = False

    def check_sentinels(self):
        self.checks += 1
        if self.fail_final and self.checks == 2:
            raise RuntimeError("Final repeat drifted")

    def verify_model(self):
        pass

    def score(self, prefix, target, single_token):
        self.requests_sent += 1
        return {"logprob": -len(target), "target_ids": [1]}


@pytest.mark.parametrize("fail_final", [False, True])
def test_complete_matrix_requires_final_stability_check(tmp_path, fail_final):
    sequences = {
        "+": {"reference": "A" * 255, "alternate": "A" * 127 + "C" + "A" * 127},
        "-": {"reference": "T" * 255, "alternate": "T" * 127 + "G" + "T" * 127},
    }
    rows = [{"id": str(i), "chrom": "3", "label": i == 0, "sequences": sequences} for i in range(10)]
    demos = [dict(rows[i], chrom="1") for i in (0, 1)]
    panels = tmp_path / "panels.json"
    panels.write_text(json.dumps({"seed": 11181, "panels": {"pilot": rows, "demonstrations": demos}}))
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "panels_sha256": hashlib.sha256(panels.read_bytes()).hexdigest(),
                "matching_windows": 12,
                "unique_windows": 12,
            }
        )
    )
    client = RecordingClient()
    client.fail_final = fail_final
    output = io.StringIO()
    if fail_final:
        with pytest.raises(RuntimeError, match="drifted"):
            run_panel(client, panels, reference, "pilot", 10, output)
    else:
        run_panel(client, panels, reference, "pilot", 10, output)
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert (records[-1]["type"] == "complete") is not fail_final
    scores = [record for record in records if record["type"] == "score"]
    assert len(scores) == 272
    assert len({(r["variant"], r["method"], r["strand"], r["mapping"], r["candidate"]) for r in scores}) == 272
    assert client.session.headers["X-data-parallel-rank"] == "0"
    client.session.close()
