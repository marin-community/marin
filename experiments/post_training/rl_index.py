# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read and write stored attempt summaries through the RL Progress applet."""

import tomllib
from pathlib import Path
from urllib.parse import urlencode

from finestore.rl.rl_progress import AttemptSummary
from pydantic import TypeAdapter

APPLET_ID = tomllib.loads(
    (Path(__file__).resolve().parents[2] / "infra/marina/apps/rl_progress/applet.toml").read_text()
)["applet_id"]
_RUNS_PATH = f"/a/{APPLET_ID}/api/runs"
_ATTEMPTS = TypeAdapter(list[AttemptSummary])


def _app(service_url: str | None) -> tuple[str, str]:
    from marina import client  # noqa: PLC0415 - read Marina dependencies at the launcher boundary

    service_url = service_url or client.DEFAULT_MARINA_URL
    details = client.marina_request(service_url, "GET", f"/api/marina/applets/{APPLET_ID}")
    url = details["authenticated_url"]
    return client.applet_origin(service_url, url), url


def list_attempts(service_url: str | None = None) -> list[AttemptSummary]:
    """Read the validated attempt index from the configured Marina service."""
    from marina import client  # noqa: PLC0415 - read Marina dependencies at the launcher boundary

    origin, _ = _app(service_url)
    response = client.marina_request(origin, "GET", _RUNS_PATH)
    return _ATTEMPTS.validate_python(response["runs"], strict=False)


def publish_result(summary: AttemptSummary, service_url: str | None = None) -> str:
    """Upsert a native attempt and return its verified dashboard URL."""
    from marina import client  # noqa: PLC0415 - read Marina dependencies at the launcher boundary

    origin, url = _app(service_url)
    served = client.marina_request(origin, "POST", _RUNS_PATH, json_body=summary.model_dump(mode="json"))
    if not isinstance(served, dict) or (served.get("run_id"), served.get("attempt_id")) != (
        summary.run_id,
        summary.attempt_id,
    ):
        raise RuntimeError(f"Published attempt {summary.run_id}/{summary.attempt_id} has a different returned identity")
    return url + "#" + urlencode({"run": summary.run_id, "attempt": summary.attempt_id})
