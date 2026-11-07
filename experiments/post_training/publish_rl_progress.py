# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish an immutable RL progress record and verify its served identity."""

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marin.rl.rl_progress import RunSummary

APPLET_ID = tomllib.loads(
    (Path(__file__).resolve().parents[2] / "infra/marina/apps/rl_progress/applet.toml").read_text()
)["applet_id"]
INSERT = """INSERT INTO rl_runs (run_id, attempt_id, record)
VALUES (:run_id, :attempt_id, CAST(:record AS jsonb))
ON CONFLICT (run_id, attempt_id) DO NOTHING"""
READ = "SELECT record FROM rl_runs WHERE run_id = :run_id AND attempt_id = :attempt_id"


@dataclass(frozen=True)
class PublicationReceipt:
    run_id: str
    attempt_id: str
    inserted: bool
    verified: bool
    url: str


def publish_result(record: dict[str, Any], service_url: str | None = None) -> PublicationReceipt:
    """Insert one immutable attempt and verify its exact readback and served identity."""
    from marina import client  # noqa: PLC0415

    service_url = service_url or client.DEFAULT_MARINA_URL
    summary = RunSummary.model_validate(record)
    encoded = json.dumps(record, allow_nan=False)
    details = client.marina_request(service_url, "GET", f"/api/marina/applets/{APPLET_ID}")
    query = client.applet_query(service_url, APPLET_ID, authenticated_url=details["authenticated_url"])
    origin = client.applet_origin(service_url, details["authenticated_url"])
    identity = {"run_id": summary.run_id, "attempt_id": summary.attempt_id}
    inserted = query(INSERT, identity | {"record": encoded})
    if inserted["row_count"] not in (0, 1):
        raise RuntimeError("Unexpected insertion count")
    readback = query(READ, identity)
    if len(readback["rows"]) != 1 or readback["rows"][0]["record"] != record:
        raise ValueError("Attempt identity already contains a different record; historical data was not overwritten")
    api = client.marina_request(origin, "GET", f"/a/{APPLET_ID}/api/runs")
    matching = [run for run in api["runs"] if (run["run_id"], run["attempt_id"]) == (summary.run_id, summary.attempt_id)]
    if len(matching) != 1:
        raise RuntimeError("Published attempt is missing or duplicated in the dashboard API")
    return PublicationReceipt(
        run_id=summary.run_id,
        attempt_id=summary.attempt_id,
        inserted=inserted["row_count"] == 1,
        verified=True,
        url=details["authenticated_url"],
    )
