# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""W&B series as flat chart rows for Grafana.

Two readers over the same public GraphQL API. `points` follows the runset pinned
by Marin's public hero-run report. `run_history` reads one named run's whole
logged history for one metric, which is what lets a step-axis panel start at step
0: finelog evicts telemetry segments once the namespace passes its storage policy,
while W&B keeps the run.
"""

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

_RUN_URL = "https://wandb.ai/{entity}/{project}/runs/{run}"
_RUN_HISTORY_SAMPLES = 2000
# W&B's own step counter. Levanter logs every training metric through
# `wandb.log(..., step=<training step>)`, so this column is the Levanter step.
_STEP_KEY = "_step"

# The projects a run named by the training dashboard can live in, searched in this
# order. The grug hero launchers default to marin_moe and marin.experiment.train
# defaults to marin. A caller that knows the project pins it and skips the search.
RUN_HISTORY_PROJECTS = ("marin_moe", "marin")

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
    """Reads the public hero-run report's runset, and any single run's history."""

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

    def _sampled_history(
        self, *, project: str, run: str, x_key: str, y_key: str, samples: int
    ) -> list[tuple[float, float]] | None:
        """Numeric (x, y) pairs from one run's sampled history, or None if it is absent.

        A point missing either key is dropped: W&B writes a null wherever a metric
        was not logged on that step. Callers decide what an absent run means.
        """
        spec = json.dumps({"keys": [x_key, y_key], "samples": samples})
        run_data = (
            self._graphql(
                _HISTORY_QUERY,
                {"entity": _ENTITY, "project": project, "run": run, "specs": [spec]},
            ).get("project")
            or {}
        ).get("run")
        if not run_data:
            return None
        histories = run_data.get("sampledHistory") or []
        pairs: list[tuple[float, float]] = []
        for point in histories[0] if histories else []:
            x_value = point.get(x_key)
            y_value = point.get(y_key)
            if isinstance(x_value, int | float) and isinstance(y_value, int | float):
                pairs.append((x_value, y_value))
        return pairs

    def points(self, chart_key: str) -> list[dict]:
        """Return one row per sampled point for a configured report chart."""
        if chart_key not in WANDB_CHARTS:
            raise ValueError(f"unknown W&B chart {chart_key!r}; configured: {sorted(WANDB_CHARTS)}")
        chart_title, metric = WANDB_CHARTS[chart_key]
        report_title, runs = self._report()
        rows: list[dict] = []
        for run in runs:
            pairs = self._sampled_history(project=_PROJECT, run=run, x_key=_X_KEY, y_key=metric, samples=_SAMPLES)
            if pairs is None:
                raise UpstreamError("wandb", f"run {run!r} not found", status_code=502)
            rows.extend(
                {
                    "chart": chart_title,
                    "run": run,
                    "tokens": tokens,
                    "value": value,
                    "report_title": report_title,
                    "report_url": _REPORT_URL,
                }
                for tokens, value in pairs
            )
        return rows

    def run_history(self, run: str, *, metric: str, project: str | None = None) -> list[dict]:
        """Return one row per sampled point of `metric` across the whole of `run`.

        `run` is the Levanter run id: marin names the W&B run after it, and
        `resume="allow"` keeps one W&B run across restarts, so this covers the run
        from step 0 however many times it was resumed. W&B samples server-side, so
        the response stays small on a long run. A run absent from every searched
        project fails loud rather than rendering as an empty panel.
        """
        projects = (project,) if project else RUN_HISTORY_PROJECTS
        for candidate in projects:
            pairs = self._sampled_history(
                project=candidate, run=run, x_key=_STEP_KEY, y_key=metric, samples=_RUN_HISTORY_SAMPLES
            )
            if pairs is None:
                continue
            run_url = _RUN_URL.format(entity=_ENTITY, project=candidate, run=run)
            return [
                {"run": run, "project": candidate, "run_url": run_url, "step": step, "value": value}
                for step, value in pairs
            ]
        raise UpstreamError("wandb", f"run {run!r} not found in {', '.join(projects)}", status_code=404)
