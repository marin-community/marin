# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a persisted Snowball loop summary without replacing historical attempts."""

import json
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import click
from marin.rl.rl_progress import RunSummary

APPLET_ID = tomllib.loads(
    (Path(__file__).resolve().parents[2] / "infra/marina/apps/rl_progress/applet.toml").read_text()
)["applet_id"]
INSERT = """INSERT INTO rl_runs (run_id, attempt_id, record)
VALUES (:run_id, :attempt_id, CAST(:record AS jsonb))
ON CONFLICT (run_id, attempt_id) DO NOTHING"""
READ = "SELECT record FROM rl_runs WHERE run_id = :run_id AND attempt_id = :attempt_id"


def publish_result(record: dict[str, Any], service_url: str | None = None) -> dict[str, Any]:
    """Insert one immutable attempt and verify its exact readback and served identity."""
    from marina import client  # noqa: PLC0415

    service_url = service_url or client.DEFAULT_MARINA_URL
    summary = RunSummary.model_validate(record)
    encoded = json.dumps(record, allow_nan=False)
    query = client.applet_query(service_url, APPLET_ID)
    details = client.marina_request(service_url, "GET", f"/api/marina/applets/{APPLET_ID}")
    location = urlparse(details["authenticated_url"])
    origin = f"{location.scheme}://{location.netloc}" if location.netloc else service_url.rstrip("/")
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
    return {
        "run_id": summary.run_id,
        "attempt_id": summary.attempt_id,
        "inserted": inserted["row_count"] == 1,
        "verified": True,
        "url": details["authenticated_url"],
    }


@click.command()
@click.argument("summary_path", required=False, type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--preset", "preset_name", help="Preset whose pinned controls to publish.")
@click.option("--controls", is_flag=True, help="Publish every pinned control from its stored native outputs.")
@click.option("--url", help="Marina origin (defaults to the Marina service).")
def main(summary_path: Path | None, preset_name: str | None, controls: bool, url: str | None) -> None:
    """Publish a persisted JSON SUMMARY_PATH or a preset's pinned controls."""
    if controls:
        if not preset_name or summary_path is not None:
            raise click.UsageError("Use --preset NAME --controls without SUMMARY_PATH")
        from experiments.post_training import snowball_rl_publication  # noqa: PLC0415

        result = snowball_rl_publication.publish_controls(preset_name, service_url=url)
        click.echo(json.dumps(result))
        if result["unserved_records"]:
            raise click.exceptions.Exit(1)
        return
    if preset_name is not None or summary_path is None:
        raise click.UsageError("Supply SUMMARY_PATH or --preset NAME --controls")
    record = json.loads(summary_path.read_text())
    click.echo(json.dumps(publish_result(record, url)))


if __name__ == "__main__":
    main()
