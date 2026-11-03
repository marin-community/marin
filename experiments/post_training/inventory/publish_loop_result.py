# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a persisted Snowball loop summary without replacing historical attempts."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import click
from marin.rl.evaluation_identity import validate_evaluation_identity
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, field_validator, model_validator

APPLET_ID = "6c971887-1e22-45b1-9dcb-dd775829df57"
DEFAULT_MARINA_URL = "https://marina.oa.dev"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost"})
INSERT = """INSERT INTO rl_runs (run_id, attempt_id, record)
VALUES (:run_id, :attempt_id, CAST(:record AS jsonb))
ON CONFLICT (run_id, attempt_id) DO NOTHING"""
READ = "SELECT record FROM rl_runs WHERE run_id = :run_id AND attempt_id = :attempt_id"
Count = Annotated[StrictInt, Field(ge=0)]
Identity = Annotated[str, Field(min_length=1)]


class EvaluationResult(BaseModel):
    model_config = ConfigDict(strict=True)
    reward: FiniteFloat
    attempted: Count
    scored: Count

    @model_validator(mode="after")
    def valid_coverage(self):
        if self.scored > self.attempted:
            raise ValueError("Scored count exceeds attempted count")
        return self


class EvaluationIdentity(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    name: Identity
    version: Identity
    task_count: Annotated[StrictInt, Field(gt=0)]
    task_set_sha256: Identity | None
    protocol_sha256: str | None
    task_ids: list[str] | None = None
    membership_sha256: str | None = None
    protocol: dict[str, Any] | None = None

    @model_validator(mode="after")
    def matching_evaluation_identity(self):
        validate_evaluation_identity(
            task_ids=self.task_ids,
            task_count=self.task_count,
            membership_sha256=self.membership_sha256,
            protocol=self.protocol,
            protocol_sha256=self.protocol_sha256,
        )
        return self


class RunSummary(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    run_id: Identity
    attempt_id: Identity
    label: Identity
    run_date: str
    date_basis: Literal["submission_acknowledged", "launch_started"]
    started_at: str
    completed_at: str | None
    elapsed_seconds: Annotated[FiniteFloat, Field(ge=0)]
    timing_scope: Identity
    status: Literal["succeeded", "failed", "running"]
    optimizer_steps: Count
    source_revision: Identity
    data_identity: Identity
    initializer_identity: Identity
    evaluation: EvaluationIdentity
    baseline: EvaluationResult | None
    final: EvaluationResult | None
    artifact_uri: str | None
    evidence: list[str]

    @field_validator("run_date", "started_at", "completed_at")
    @classmethod
    def utc_timestamp(cls, value):
        if value is not None:
            parsed = datetime.fromisoformat(value)
            if parsed.utcoffset() != timedelta(0):
                raise ValueError("Run timestamps must explicitly use UTC")
        return value

    @model_validator(mode="after")
    def valid_outcome(self):
        if self.status != "running" and self.completed_at is None:
            raise ValueError("Terminal attempts require a completion timestamp")
        if self.final is not None and (self.baseline is None or self.elapsed_seconds <= 0):
            raise ValueError("A final measurement requires a baseline and positive elapsed time")
        for phase in (self.baseline, self.final):
            if phase is not None and phase.attempted > self.evaluation.task_count:
                raise ValueError("Attempted count exceeds the fixed evaluation cohort")
        if self.status == "succeeded" and any(
            phase is None or phase.attempted != self.evaluation.task_count or phase.scored != self.evaluation.task_count
            for phase in (self.baseline, self.final)
        ):
            raise ValueError("Successful attempts require complete baseline and final scoring")
        return self


def marina_request(service_url: str, method: str, path: str, *, json_body: object | None = None) -> object | None:
    """Load the deployment client only when publishing, outside ML graph construction."""
    try:
        from marina.client import marina_request as request  # noqa: PLC0415  # optional deployment dependency
    except ModuleNotFoundError as error:
        if error.name not in {"marina", "marina.client"}:
            raise
        raise RuntimeError(
            "Publishing loop results requires the Marina client; run publication from the repository workspace."
        ) from error
    return request(service_url, method, path, json_body=json_body)


def publish_result(record: dict[str, Any], service_url: str = DEFAULT_MARINA_URL) -> dict[str, Any]:
    """Insert one immutable attempt, then verify its stored JSON and serving coverage."""
    summary = RunSummary.model_validate(record)
    encoded = json.dumps(record, allow_nan=False)
    parsed_service = urlparse(service_url)
    if service_url.rstrip("/") != DEFAULT_MARINA_URL and not (
        parsed_service.scheme == "http" and parsed_service.hostname in LOCAL_HOSTS and not parsed_service.path
    ):
        raise ValueError("Use the maintained Marina origin or an explicit loopback preview origin")
    details = marina_request(service_url, "GET", f"/api/marina/applets/{APPLET_ID}")
    assert isinstance(details, dict), "Marina metadata response must be an object"
    location = urlparse(details["authenticated_url"])
    origin = f"{location.scheme}://{location.netloc}" if location.netloc else service_url.rstrip("/")
    query_path = f"/a/{APPLET_ID}/query"
    identity = {"run_id": summary.run_id, "attempt_id": summary.attempt_id}
    inserted = marina_request(
        origin,
        "POST",
        query_path,
        json_body={"sql": INSERT, "parameters": identity | {"record": encoded}},
    )
    assert isinstance(inserted, dict), "Marina insertion response must be an object"
    if inserted["row_count"] not in (0, 1):
        raise RuntimeError("Unexpected insertion count")
    readback = marina_request(origin, "POST", query_path, json_body={"sql": READ, "parameters": identity})
    assert isinstance(readback, dict), "Marina readback response must be an object"
    if len(readback["rows"]) != 1 or readback["rows"][0]["record"] != record:
        raise ValueError("Attempt identity already contains a different record; historical data was not overwritten")
    api = marina_request(origin, "GET", f"/a/{APPLET_ID}/api/runs")
    assert isinstance(api, dict), "Dashboard response must be an object"
    matching = [run for run in api["runs"] if (run["run_id"], run["attempt_id"]) == (summary.run_id, summary.attempt_id)]
    if len(matching) != 1:
        raise RuntimeError("Published attempt is missing or duplicated in the dashboard API")
    served = matching[0]
    for field in ("status", "optimizer_steps", "baseline", "final"):
        if served[field] != record[field]:
            raise RuntimeError(f"Dashboard API disagrees with stored {field}")
    complete = all(
        phase is not None and phase.scored == phase.attempted == summary.evaluation.task_count
        for phase in (summary.baseline, summary.final)
    )
    if served["partial_coverage"] != (not complete):
        raise RuntimeError("Dashboard API disagrees with actual evaluation coverage")
    if (summary.evaluation.protocol_sha256 is None or summary.evaluation.task_set_sha256 is None) and (
        served["comparison_key"] is not None or not served["provisional"] or served["pareto_eligible"]
    ):
        raise RuntimeError("Unknown runtime evaluation identity was incorrectly made rankable")
    return {
        "run_id": summary.run_id,
        "attempt_id": summary.attempt_id,
        "inserted": inserted["row_count"] == 1,
        "verified": True,
        "url": details["authenticated_url"],
    }


@click.command()
@click.argument("summary_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--url", default=DEFAULT_MARINA_URL, show_default=True, help="Marina origin or loopback preview origin.")
def main(summary_path: Path, url: str) -> None:
    """Publish the persisted JSON SUMMARY_PATH and print an exact-readback receipt."""
    record = json.loads(summary_path.read_text())
    click.echo(json.dumps(publish_result(record, url)))


if __name__ == "__main__":
    main()
