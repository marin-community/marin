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
# The projected finish extrapolates the step rate over this many most recent steps: about
# four hours of the hero at 15 s a step. A whole-life rate lags a throughput change for
# days and keeps every past outage in the average; a trailing window sheds an outage once
# the run has trained through it. Sampled by step, so the window stays dense however long
# the run is.
_RATE_WINDOW_STEPS = 1_000
_RATE_WINDOW_SAMPLES = 50

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
    run(name: $run) { state branchPoint { step } sampledHistory(specs: $specs) }
  }
}
"""

# `summaryMetrics` carries the last value logged for every key, `_runtime` among them.
# Reading the clocks therefore costs one small request, not a history download.
_ACTIVITY_QUERY = """
query RunActivity($entity: String!, $project: String!, $run: String!) {
  project(entityName: $entity, name: $project) {
    run(name: $run) { state createdAt heartbeatAt summaryMetrics branchPoint { step } }
  }
}
"""


def _epoch_seconds(stamp: str) -> float:
    """Epoch seconds for a W&B RFC-3339 stamp, whose zone is always `Z`."""
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


class _HistorySpec(NamedTuple):
    keys: tuple[str, ...]
    samples: int
    min_step: int | None = None


class _SampledHistories(NamedTuple):
    """One point list per requested spec, and the run's fork boundary."""

    points: list[list[dict[str, float]]]
    branch_step: int | None


class _HistoryBaseline(NamedTuple):
    reference_tps: float
    tokens_baseline: float


def _history_baseline(points: list[dict[str, float]]) -> _HistoryBaseline | None:
    """The reference token rate and the token count a W&B run started from.

    The mean of the sampled rate is the reference speed -- a mean over history, not
    the summary's last-step value, so a checkpoint or eval step cannot skew it (see
    `_TPS_KEY`).

    The token baseline is the cumulative token count before the run's first step,
    which a run resumed under a fresh id from a mid-schedule checkpoint carries in
    from the checkpoint (levanter derives `total_tokens` from the global step). It
    is reconstructed rather than read off the earliest sample, because that sample
    is logged *after* the first step and so already includes one batch: with a
    constant batch, `total_tokens` is proportional to `step + 1`, so the
    pre-first-step count is `first_tokens * first_step / (first_step + 1)` -- zero
    for a run started from scratch, the inherited count for a resumed one.
    None before the first logged step.
    """
    if not points:
        return None
    reference_tps = sum(point[_TPS_KEY] for point in points) / len(points)
    first = min(points, key=lambda point: point[_STEP_KEY])
    step, tokens = first[_STEP_KEY], first[_TOTAL_TOKENS_KEY]
    return _HistoryBaseline(reference_tps=reference_tps, tokens_baseline=tokens * step / (step + 1))


def _projected_finish_ms(window: list[dict[str, float]]) -> int | None:
    """Epoch milliseconds when the run reaches its stop step at the window's step rate.

    The rate runs from the window's first logged point to its last, both read from
    W&B's per-step stamps, and carries forward from the last one to the stop step
    that point's `_step / run_progress` recovers. None until the window spans two
    distinct steps.
    """
    if not window:
        return None
    first = min(window, key=lambda point: point[_STEP_KEY])
    last = max(window, key=lambda point: point[_STEP_KEY])
    steps = last[_STEP_KEY] - first[_STEP_KEY]
    seconds = last[_TIMESTAMP_KEY] - first[_TIMESTAMP_KEY]
    progress = last[_PROGRESS_KEY]
    if progress <= 0 or steps <= 0 or seconds <= 0:
        return None
    steps_remaining = last[_STEP_KEY] / progress - last[_STEP_KEY]
    return round((last[_TIMESTAMP_KEY] + steps_remaining * seconds / steps) * 1000)


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

    def _sampled_run_histories(self, *, project: str, run: str, specs: list[_HistorySpec]) -> _SampledHistories | None:
        """Numeric history per spec and the fork boundary, or None if the run is absent.

        One request serves every spec. Each point carries every key in its spec; a
        point missing any of them is dropped, since W&B writes a null wherever a metric
        was not logged on that step. Callers decide what an absent run means.
        """
        encoded = [
            json.dumps(
                {
                    "keys": list(spec.keys),
                    "samples": spec.samples,
                    **({"minStep": spec.min_step} if spec.min_step is not None else {}),
                }
            )
            for spec in specs
        ]
        run_data = (
            self._graphql(
                _HISTORY_QUERY,
                {"entity": _ENTITY, "project": project, "run": run, "specs": encoded},
            ).get("project")
            or {}
        ).get("run")
        if not run_data:
            return None
        histories = run_data.get("sampledHistory") or []
        points: list[list[dict[str, float]]] = []
        for index, spec in enumerate(specs):
            kept = []
            for point in histories[index] if index < len(histories) else []:
                values = {key: point.get(key) for key in spec.keys}
                if all(isinstance(value, int | float) for value in values.values()):
                    kept.append(values)
            points.append(kept)
        branch_point = run_data.get("branchPoint")
        return _SampledHistories(points, int(branch_point["step"]) if branch_point else None)

    def _sampled_plot_points(
        self, *, project: str, run: str, keys: tuple[str, ...], samples: int
    ) -> list[dict[str, float]] | None:
        """Return step-ordered plot points, preserving detail in a fork's child segment."""
        history = self._sampled_run_histories(project=project, run=run, specs=[_HistorySpec(keys, samples)])
        if history is None:
            return None
        (points,) = history.points
        if history.branch_step is not None:
            # Inherited history can consume almost the entire sample budget.
            # Sample the child's segment separately, retaining the parent prefix.
            child = self._sampled_run_histories(
                project=project, run=run, specs=[_HistorySpec(keys, samples, history.branch_step + 1)]
            )
            if child is None:
                raise UpstreamError("wandb", f"run {run!r} disappeared while reading history", status_code=502)
            points = [point for point in points if point[_STEP_KEY] <= history.branch_step] + child.points[0]
        return sorted(points, key=lambda point: point[_STEP_KEY])

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
            points = self._sampled_plot_points(
                project=_PROJECT, run=run, keys=(_STEP_KEY, _TOTAL_TOKENS_KEY, metric), samples=_SAMPLES
            )
            if points is None:
                raise UpstreamError("wandb", f"run {run!r} not found", status_code=502)
            rows.extend(
                {
                    "chart": chart_title,
                    "run": run,
                    "tokens": point[_TOTAL_TOKENS_KEY],
                    "value": point[metric],
                    "report_title": report_title,
                    "report_url": _REPORT_URL,
                }
                for point in points
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
            points = self._sampled_plot_points(
                project=candidate, run=run, keys=(_STEP_KEY, metric), samples=_RUN_HISTORY_SAMPLES
            )
            if points is None:
                return None
            run_url = _RUN_URL.format(entity=_ENTITY, project=candidate, run=run)
            return [
                {"run": run, "project": candidate, "run_url": run_url, "step": point[_STEP_KEY], "value": point[metric]}
                for point in points
            ]

        return self._search_projects(run, project, read)

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

        `projected_finish_ms` is the epoch millisecond at which a running run reaches
        its stop step at its recent step rate (see `_projected_finish_ms`), sampled over
        the last `_RATE_WINDOW_STEPS` steps. Checkpoints, evals, and restarts inside the
        window slow the rate; an outage the run has trained past no longer does. The
        window never reaches into a fork's inherited history. A stall or a replay of
        already-logged steps logs nothing, so it moves the date only once the run logs
        a new step. Null for a run that is not running, until the window spans two
        steps, and for a run that does not log `run_progress`.
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
            branch_point = run_data.get("branchPoint")
            # A fork's sampled history includes its parent's; start after the branch.
            own_min_step = int(branch_point["step"]) + 1 if branch_point else None
            specs = [_HistorySpec((_STEP_KEY, _TOTAL_TOKENS_KEY, _TPS_KEY), _TPS_SAMPLES, own_min_step)]
            step = summary.get(_STEP_KEY)
            projecting = run_data.get("state") == "running" and isinstance(step, int | float)
            if projecting:
                window_start = max(int(step) - _RATE_WINDOW_STEPS, own_min_step or 0)
                specs.append(
                    _HistorySpec((_STEP_KEY, _TIMESTAMP_KEY, _PROGRESS_KEY), _RATE_WINDOW_SAMPLES, window_start)
                )
            histories = self._sampled_run_histories(project=candidate, run=run, specs=specs)
            if histories is None:
                raise UpstreamError("wandb", f"run {run!r} disappeared while reading history", status_code=502)
            baseline = _history_baseline(histories.points[0])
            reference_tps = baseline.reference_tps if baseline else None
            tokens_since_start = tokens_seen - baseline.tokens_baseline if tokens_seen is not None and baseline else None
            efficiency = (
                tokens_since_start / (reference_tps * wall)
                if tokens_since_start is not None and tokens_since_start > 0 and reference_tps and wall > 0
                else None
            )
            projected_finish_ms = _projected_finish_ms(histories.points[1]) if projecting else None
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
