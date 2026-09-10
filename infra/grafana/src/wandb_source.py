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
from typing import NamedTuple

import httpx
from errors import UpstreamError
from graphql_source import graphql_data

_GRAPHQL_URL = "https://api.wandb.ai/graphql"
_ENTITY = "marin-community"
_PROJECT = "marin_moe"
_REPORT_VIEW_ID = "VmlldzoxNzc2MDM5Ng=="
_REPORT_URL = (
    "https://wandb.ai/marin-community/marin_moe/reports/535B-A23B-18T-Token-Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng"
)
_TOTAL_TOKENS_KEY = "throughput/total_tokens"
_SAMPLES = 800

WANDB_CHARTS = {
    "train-loss": ("Train cross-entropy loss", "train/cross_entropy_loss"),
    "paloma-macro-loss": ("Paloma macro loss (dropless)", "eval_dropless/paloma/macro_loss"),
    "mfu": ("MFU (%)", "throughput/mfu"),
}

_RUN_URL = "https://wandb.ai/{entity}/{project}/runs/{run}"
_RUN_HISTORY_SAMPLES = 2000
# W&B's own step counter. Levanter logs every training metric through
# `wandb.log(..., step=<training step>)`, so this column is the Levanter step.
_STEP_KEY = "_step"
# W&B's own wall-clock stamp on every logged point, in epoch seconds.
_TIMESTAMP_KEY = "_timestamp"
# Schedule progress, logged with every step by `levanter.callbacks.log_step_info`. The
# grug trainers log the global step over the step the run stops at, so `_step` over
# it is the stop step exactly. Levanter's own trainer logs examples through step + 1
# over the schedule's total, which with a constant batch is (step + 1) over the stop
# step: the recovered stop step is then low by a part in `_step`, and a batch ramp
# makes it the example-weighted equivalent, which a step-rate extrapolation reads as
# an approximation either way.
_PROGRESS_KEY = "run_progress"

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


class _HistoryBaseline(NamedTuple):
    reference_tps: float
    tokens_baseline: float
    first_step: float
    first_timestamp: float


def _projected_finish_ms(
    *, step: float, progress: float, baseline: _HistoryBaseline, heartbeat_seconds: float
) -> int | None:
    """Epoch milliseconds when the run reaches `step / progress` at its own step rate.

    The rate is `step` less the first sampled step, over the heartbeat less the first
    sample's stamp. None until the run has advanced past its first sample.
    """
    steps_since_first = step - baseline.first_step
    seconds_since_first = heartbeat_seconds - baseline.first_timestamp
    if progress <= 0 or steps_since_first <= 0 or seconds_since_first <= 0:
        return None
    steps_remaining = step / progress - step
    return round((heartbeat_seconds + steps_remaining * seconds_since_first / steps_since_first) * 1000)


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

    def _sampled_points(
        self, *, project: str, run: str, keys: tuple[str, ...], samples: int
    ) -> list[dict[str, float]] | None:
        """Numeric points from one run's sampled history, or None if the run is absent.

        Each point carries every key in `keys`; a point missing any of them is dropped,
        since W&B writes a null wherever a metric was not logged on that step. Callers
        decide what an absent run means.
        """
        spec = json.dumps({"keys": list(keys), "samples": samples})
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
        points: list[dict[str, float]] = []
        for point in histories[0] if histories else []:
            values = {key: point.get(key) for key in keys}
            if all(isinstance(value, int | float) for value in values.values()):
                points.append(values)
        return points

    def _sampled_history(
        self, *, project: str, run: str, x_key: str, y_key: str, samples: int
    ) -> list[tuple[float, float]] | None:
        """Numeric (x, y) pairs from one run's sampled history, or None if it is absent."""
        points = self._sampled_points(project=project, run=run, keys=(x_key, y_key), samples=samples)
        if points is None:
            return None
        return [(point[x_key], point[y_key]) for point in points]

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

    def _history_baseline(self, *, project: str, run: str) -> _HistoryBaseline | None:
        """The reference token rate and where this W&B run's own work begins.

        All from one sampled history. The mean of the rate is the reference speed
        -- a mean over history, not the summary's last-step value, so a checkpoint or
        eval step cannot skew it (see `_TPS_KEY`).

        The token baseline is the cumulative token count before this run's first step,
        which a run resumed under a fresh id from a mid-schedule checkpoint carries in
        from the checkpoint (levanter derives `total_tokens` from the global step). It
        is reconstructed rather than read off the earliest sample, because that sample
        is logged *after* the first step and so already includes one batch: with a
        constant batch, `total_tokens` is proportional to `step + 1`, so the
        pre-first-step count is `first_tokens * first_step / (first_step + 1)` -- zero
        for a run started from scratch, the inherited count for a resumed one.

        The first sample's step and stamp anchor the step rate: measured from there,
        the steps a resumed run inherited and the time before its first step both drop
        out. None before the first logged step.
        """
        points = self._sampled_points(
            project=project,
            run=run,
            keys=(_STEP_KEY, _TIMESTAMP_KEY, _TOTAL_TOKENS_KEY, _TPS_KEY),
            samples=_TPS_SAMPLES,
        )
        if not points:
            return None
        reference_tps = sum(point[_TPS_KEY] for point in points) / len(points)
        first = min(points, key=lambda point: point[_STEP_KEY])
        step, tokens = first[_STEP_KEY], first[_TOTAL_TOKENS_KEY]
        return _HistoryBaseline(
            reference_tps=reference_tps,
            tokens_baseline=tokens * step / (step + 1),
            first_step=step,
            first_timestamp=first[_TIMESTAMP_KEY],
        )

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

        `projected_finish_ms` is the epoch millisecond at which the run reaches its
        stop step at its own step rate: the steps between the first sampled point and
        the summary's `_step`, over the wall clock between that point and the last
        heartbeat, carried forward over `_step / run_progress - _step` steps to go.
        Measuring from the first sample means a run resumed from a checkpoint is not
        credited with the steps it inherited, and every restart, checkpoint, and eval
        since then slows the rate. The window is the run's whole life under this id.
        Null until the run has advanced past its first sample, and for a run that
        does not log `run_progress`.
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
            heartbeat_seconds = _epoch_seconds(run_data["heartbeatAt"])
            wall = heartbeat_seconds - _epoch_seconds(run_data["createdAt"])
            tokens_seen = summary.get(_TOTAL_TOKENS_KEY)
            tokens_seen = float(tokens_seen) if isinstance(tokens_seen, int | float) else None
            baseline = self._history_baseline(project=candidate, run=run)
            reference_tps = baseline.reference_tps if baseline else None
            tokens_since_start = tokens_seen - baseline.tokens_baseline if tokens_seen is not None and baseline else None
            efficiency = (
                tokens_since_start / (reference_tps * wall)
                if tokens_since_start is not None and tokens_since_start > 0 and reference_tps and wall > 0
                else None
            )
            step, progress = summary.get(_STEP_KEY), summary.get(_PROGRESS_KEY)
            projected_finish_ms = (
                _projected_finish_ms(
                    step=step,
                    progress=progress,
                    baseline=baseline,
                    heartbeat_seconds=heartbeat_seconds,
                )
                if baseline and isinstance(step, int | float) and isinstance(progress, int | float)
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
                    "projected_finish_ms": projected_finish_ms,
                }
            ]

        return self._search_projects(run, project, read)
