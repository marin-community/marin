# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish an RL progress record and verify its served identity."""

import json
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from marin.rl.rl_progress import RunSummary

APPLET_ID = tomllib.loads(
    (Path(__file__).resolve().parents[2] / "infra/marina/apps/rl_progress/applet.toml").read_text()
)["applet_id"]
INSERT = """INSERT INTO rl_runs (run_id, attempt_id, record)
VALUES (:run_id, :attempt_id, CAST(:record AS jsonb))
ON CONFLICT (run_id, attempt_id) DO UPDATE SET record = jsonb_set(
    EXCLUDED.record, '{issue}',
    COALESCE(NULLIF(EXCLUDED.record -> 'issue', 'null'::jsonb), rl_runs.record -> 'issue', 'null'::jsonb))"""


def publish_result(record: dict[str, Any], service_url: str | None = None) -> str:
    """Upsert one native attempt and return its verified dashboard URL."""
    from marina import client  # noqa: PLC0415

    service_url = service_url or client.DEFAULT_MARINA_URL
    summary = RunSummary.model_validate(record)
    encoded = json.dumps(record, allow_nan=False)
    details = client.marina_request(service_url, "GET", f"/api/marina/applets/{APPLET_ID}")
    query = client.applet_query(service_url, APPLET_ID)
    origin = client.applet_origin(service_url, details["authenticated_url"])
    identity = {"run_id": summary.run_id, "attempt_id": summary.attempt_id}
    query(INSERT, identity | {"record": encoded})
    api = client.marina_request(origin, "GET", f"/a/{APPLET_ID}/api/runs")
    matching = [run for run in api["runs"] if (run["run_id"], run["attempt_id"]) == (summary.run_id, summary.attempt_id)]
    if len(matching) != 1:
        skipped = [
            row["error"]
            for row in api.get("skipped_records", [])
            if all(row[key] == value for key, value in identity.items())
        ]
        raise RuntimeError(
            f"Published attempt {summary.run_id}/{summary.attempt_id} is not served exactly once: {skipped}"
        )
    return details["authenticated_url"] + "#" + urlencode({"run": summary.run_id, "attempt": summary.attempt_id})
