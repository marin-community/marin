# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""W&B series as flat chart rows for Grafana.

Three readers over the same public GraphQL API. `points` follows the runset pinned
by Marin's public hero-run report. `run_history` reads one named run's whole
logged history for one metric, which is what lets a step-axis panel start at step
0: finelog evicts telemetry segments once the namespace passes its storage policy,
while W&B keeps the run. `run_activity` reads the same run's clocks, for the same
reason: a run's total active time spans every attempt it ever had, and finelog
retains only a window of them.
"""

import json
from collections.abc import Callable
from datetime import datetime

import httpx
from errors import UpstreamError
from graphql_source import graphql_data

_GRAPHQL_URL = "https://api.wandb.ai/graphql"
_ENTITY = "marin-community"
_PROJECT = "marin_moe"
_REPORT_VIEW_ID = "VmlldzoxNzM1OTMxMQ=="
_REPORT_URL = "https://wandb.ai/marin-community/marin_moe/reports/67B-A2B-MoE-on-10T-tokens--VmlldzoxNzM1OTMxMQ"
_TOTAL_TOKENS_KEY = "throughput/total_tokens"
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

# Reference speed for progress efficiency. `summaryMetrics` carries only the last
# step's `throughput/tokens_per_second`, which a checkpoint or eval step drives
# ~15x low, so the mean of a sampled history is used instead: it is stable across a
# single bad step and matches the status strip's own AVG(tokens/s) tile.
_TPS_KEY = "throughput/tokens_per_second"
_TPS_SAMPLES = 500

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

# `summaryMetrics` carries the last value logged for every key, `_runtime` among them.
# Reading the clocks therefore costs one small request, not a history download.
_ACTIVITY_QUERY = """
query RunActivity($entity: String!, $project: String!, $run: String!) {
  project(entityName: $entity, name: $project) {
    run(name: $run) { state createdAt heartbeatAt summaryMetrics }
  }
}
"""


def _epoch_seconds(stamp: str) -> float:
    """Epoch seconds for a W&B RFC-3339 stamp, whose zone is always `Z`."""
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


class WandbSource:
    """Reads the public hero-run report's runset, and any single run's history and clocks."""

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

    def _search_projects(self, run: str, project: str | None, read: Callable[[str], list[dict] | None]) -> list[dict]:
        """Return the first non-empty `read(candidate)` over the projects that may hold `run`.

        A run in none of them fails loud rather than rendering as an empty panel.
        """
        projects = (project,) if project else RUN_HISTORY_PROJECTS
        for candidate in projects:
            rows = read(candidate)
            if rows is not None:
                return rows
        raise UpstreamError("wandb", f"run {run!r} not found in {', '.join(projects)}", status_code=404)

    def points(self, chart_key: str) -> list[dict]:
        """Return one row per sampled point for a configured report chart."""
        if chart_key not in WANDB_CHARTS:
            raise ValueError(f"unknown W&B chart {chart_key!r}; configured: {sorted(WANDB_CHARTS)}")
        chart_title, metric = WANDB_CHARTS[chart_key]
        report_title, runs = self._report()
        rows: list[dict] = []
        for run in runs:
            pairs = self._sampled_history(
                project=_PROJECT, run=run, x_key=_TOTAL_TOKENS_KEY, y_key=metric, samples=_SAMPLES
            )
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
        the response stays small on a long run.
        """

        def read(candidate: str) -> list[dict] | None:
            pairs = self._sampled_history(
                project=candidate, run=run, x_key=_STEP_KEY, y_key=metric, samples=_RUN_HISTORY_SAMPLES
            )
            if pairs is None:
                return None
            run_url = _RUN_URL.format(entity=_ENTITY, project=candidate, run=run)
            return [
                {"run": run, "project": candidate, "run_url": run_url, "step": step, "value": value}
                for step, value in pairs
            ]

        return self._search_projects(run, project, read)

    def _reference_rate_and_token_baseline(self, *, project: str, run: str) -> tuple[float, float] | tuple[None, None]:
        """Mean per-step token rate and the run's starting cumulative token count.

        Both come from one sampled history keyed on cumulative tokens. The mean of the
        rate is the reference speed -- a mean over history, not the summary's last-step
        value, so a checkpoint or eval step cannot skew it (see `_TPS_KEY`). The
        smallest cumulative-token value is where this W&B run began: levanter derives
        `total_tokens` from the global step, so a run resumed under a fresh id from a
        mid-schedule checkpoint starts at a nonzero count, and that baseline is
        subtracted before dividing by wall time. `(None, None)` if the run has logged
        no token rate yet.
        """
        pairs = self._sampled_history(
            project=project, run=run, x_key=_TOTAL_TOKENS_KEY, y_key=_TPS_KEY, samples=_TPS_SAMPLES
        )
        if not pairs:
            return None, None
        rates = [rate for _, rate in pairs]
        baseline = min(tokens for tokens, _ in pairs)
        return sum(rates) / len(rates), baseline

    def run_activity(self, run: str, *, project: str | None = None) -> list[dict]:
        """Return one row of active time, wall-clock time, and progress efficiency for `run`.

        W&B's `_runtime` counts the seconds a process was alive: `resume="allow"`
        restores it at each restart, so the wait between two attempts never enters
        it. That makes it the run's active execution time across every attempt, and
        unlike a `telemetry_v1` scan it does not stop where segment eviction does.
        Wall time runs from the run's creation to its last heartbeat, thus the
        remainder is downtime and the ratio is the share of the run that ran.

        Progress efficiency is `tokens_since_start / (reference_tps * wall)`: the
        fraction of an ideal run that held its steady token rate from creation with no
        downtime. It counts tokens this W&B run produced -- cumulative tokens minus the
        run's starting count -- so a run resumed under a fresh id from a mid-schedule
        checkpoint is not credited the tokens it inherited. It is stricter than active
        share, which sees only downtime: this also counts the throughput lost to
        checkpoints, evals, and steps redone after a rollback. A run that has logged
        nothing yet reports nulls rather than zeros.
        """

        def read(candidate: str) -> list[dict] | None:
            run_data = (
                self._graphql(
                    _ACTIVITY_QUERY,
                    {"entity": _ENTITY, "project": candidate, "run": run},
                ).get("project")
                or {}
            ).get("run")
            if not run_data:
                return None
            summary = json.loads(run_data.get("summaryMetrics") or "{}")
            active = summary.get("_runtime")
            active = float(active) if isinstance(active, int | float) else None
            wall = _epoch_seconds(run_data["heartbeatAt"]) - _epoch_seconds(run_data["createdAt"])
            tokens_seen = summary.get(_TOTAL_TOKENS_KEY)
            tokens_seen = float(tokens_seen) if isinstance(tokens_seen, int | float) else None
            reference_tps, tokens_baseline = self._reference_rate_and_token_baseline(project=candidate, run=run)
            tokens_since_start = (
                tokens_seen - tokens_baseline if tokens_seen is not None and tokens_baseline is not None else None
            )
            efficiency = (
                tokens_since_start / (reference_tps * wall)
                if tokens_since_start is not None and tokens_since_start > 0 and reference_tps and wall > 0
                else None
            )
            return [
                {
                    "run": run,
                    "project": candidate,
                    "run_url": _RUN_URL.format(entity=_ENTITY, project=candidate, run=run),
                    "state": run_data.get("state"),
                    "active_seconds": active,
                    "wall_seconds": wall,
                    "downtime_seconds": None if active is None else wall - active,
                    "active_share": active / wall if active is not None and wall > 0 else None,
                    "reference_tps": reference_tps,
                    "progress_efficiency": efficiency,
                }
            ]

        return self._search_projects(run, project, read)
