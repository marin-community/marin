# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public W&B report series as flat chart rows for Grafana."""

import json

import httpx
from errors import UpstreamError
from graphql_source import graphql_data

_GRAPHQL_URL = "https://api.wandb.ai/graphql"
_ENTITY = "marin-community"
_PROJECT = "marin_moe"
_REPORT_VIEW_ID = "VmlldzoxNzM1OTMxMQ=="
_REPORT_URL = "https://wandb.ai/marin-community/marin_moe/reports/67B-A2B-MoE-on-10T-tokens--VmlldzoxNzM1OTMxMQ"
_X_KEY = "throughput/total_tokens"
_SAMPLES = 800

WANDB_CHARTS = {
    "train-loss": ("Train cross-entropy loss", "train/cross_entropy_loss"),
    "paloma-macro-loss": ("Paloma macro loss", "eval/paloma/macro_loss"),
    "mfu": ("MFU (%)", "throughput/mfu"),
}

_REPORT_QUERY = """
query Report($id: ID!) {
  view(id: $id) { displayName spec }
}
"""

_HISTORY_QUERY = """
query RunSampledHistory($entity: String!, $project: String!, $run: String!, $specs: [JSONString!]!) {
  project(entityName: $entity, name: $project) {
    run(name: $run) { state sampledHistory(specs: $specs) }
  }
}
"""


class WandbSource:
    """Reads the runset pinned by Marin's public hero-run report."""

    def __init__(self, *, timeout: float) -> None:
        self._client = httpx.Client(timeout=timeout, headers={"content-type": "application/json"})

    def _graphql(self, query: str, variables: dict) -> dict:
        return graphql_data(
            self._client,
            source="wandb",
            url=_GRAPHQL_URL,
            query=query,
            variables=variables,
        )

    def _report(self) -> tuple[str, list[str]]:
        view = self._graphql(_REPORT_QUERY, {"id": _REPORT_VIEW_ID}).get("view") or {}
        if not view.get("spec"):
            raise UpstreamError("wandb", "report not found", status_code=502)
        spec = json.loads(view["spec"])
        grid = next((block for block in spec.get("blocks", []) if block.get("type") == "panel-grid"), None)
        runsets = ((grid or {}).get("metadata") or {}).get("runSets") or []
        runs = ((runsets[0] if runsets else {}).get("selections") or {}).get("tree") or []
        if not runs:
            raise UpstreamError("wandb", "report pins no runs", status_code=502)
        return view.get("displayName") or "W&B report", runs

    def points(self, chart_key: str) -> list[dict]:
        """Return one row per sampled point for a configured report chart."""
        if chart_key not in WANDB_CHARTS:
            raise ValueError(f"unknown W&B chart {chart_key!r}; configured: {sorted(WANDB_CHARTS)}")
        chart_title, metric = WANDB_CHARTS[chart_key]
        report_title, runs = self._report()
        spec = json.dumps({"keys": [_X_KEY, metric], "samples": _SAMPLES})
        rows: list[dict] = []
        for run in runs:
            project = (
                self._graphql(
                    _HISTORY_QUERY,
                    {"entity": _ENTITY, "project": _PROJECT, "run": run, "specs": [spec]},
                ).get("project")
                or {}
            )
            run_data = project.get("run") or {}
            if not run_data:
                raise UpstreamError("wandb", f"run {run!r} not found", status_code=502)
            histories = run_data.get("sampledHistory") or []
            for point in histories[0] if histories else []:
                tokens = point.get(_X_KEY)
                value = point.get(metric)
                if isinstance(tokens, int | float) and isinstance(value, int | float):
                    rows.append(
                        {
                            "chart": chart_title,
                            "run": run,
                            "run_state": run_data.get("state") or "unknown",
                            "tokens": tokens,
                            "value": value,
                            "report_title": report_title,
                            "report_url": _REPORT_URL,
                        }
                    )
        return rows
