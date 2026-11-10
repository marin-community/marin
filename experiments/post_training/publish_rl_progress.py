# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish an RL progress record and verify its served identity."""

import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from marin.rl.rl_progress import RunSummary

APPLET_ID = tomllib.loads(
    (Path(__file__).resolve().parents[2] / "infra/marina/apps/rl_progress/applet.toml").read_text()
)["applet_id"]


def publish_result(record: dict[str, Any], service_url: str | None = None) -> str:
    """Upsert one native attempt and return its verified dashboard URL."""
    from marina import client  # noqa: PLC0415

    service_url = service_url or client.DEFAULT_MARINA_URL
    summary = RunSummary.model_validate(record)
    details = client.marina_request(service_url, "GET", f"/api/marina/applets/{APPLET_ID}")
    origin = client.applet_origin(service_url, details["authenticated_url"])
    served = RunSummary.model_validate(
        client.marina_request(origin, "POST", f"/a/{APPLET_ID}/api/runs", json_body=record)
    )
    if (served.run_id, served.attempt_id) != (summary.run_id, summary.attempt_id):
        raise RuntimeError(f"Published attempt {summary.run_id}/{summary.attempt_id} has a different returned identity")
    return details["authenticated_url"] + "#" + urlencode({"run": summary.run_id, "attempt": summary.attempt_id})
