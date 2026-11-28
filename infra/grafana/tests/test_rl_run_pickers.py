# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


import re
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

import pytest
from conftest import (
    dashboard_panels,
    resolved_query_params,
)
from dashboard_stitch import stitch_all
from rl_run_fixtures import (
    _COLUMNS,
    _NOW_MS,
    _WINDOW_START_MS,
    ASYNC_RUN_ID,
    CLUSTER,
    DASHBOARDS,
    DEFAULT_EXECUTION_UID,
    JOB_ID,
    NODES,
    RUN_ID,
    UNSTAMPED_RUN_ID,
    WINDOW_START,
    _dashboard,
    _panel_sql,
    _recent_runs_panel,
    _request_rows,
    _row,
)


def test_every_timeseries_panel_declares_the_columns_its_projection_returns(rl_run_store) -> None:
    """A panel is read through its declared columns, so executing its SQL cannot see a mistake there."""
    dashboard = stitch_all(DASHBOARDS, DASHBOARDS / "panels")["rl_runs.json"]

    for panel in dashboard_panels(dashboard["panels"]):
        if panel.get("type") != "timeseries":
            continue
        for target in panel["targets"]:
            if target["url"] not in ("/v1/rl/overview", "/v1/rl/gpu"):
                continue
            declared = {column["selector"]: column["type"] for column in target["columns"]}
            rl_run_store.execute(_panel_sql(panel["title"]))
            selected = {column[0] for column in rl_run_store.description}

            assert (
                set(declared) == selected
            ), f"{panel['title']}: declares {sorted(declared)}, SQL returns {sorted(selected)}"
            assert "number" in declared.values(), f"{panel['title']}: no numeric column to plot"


def test_every_panel_has_a_distinct_title_id_and_slot() -> None:
    # A duplicated panel renders twice and shares an id, and a test that looks panels up by
    # title cannot see it: the lookup keeps one and the dashboard keeps both.
    dashboard = stitch_all(DASHBOARDS, DASHBOARDS / "panels")["rl_runs.json"]
    panels = list(dashboard_panels(dashboard["panels"]))

    titles = [panel["title"] for panel in panels]
    assert len(titles) == len(set(titles)), titles
    ids = [panel["id"] for panel in panels]
    assert len(ids) == len(set(ids)), ids
    for group in (dashboard["panels"], *(panel["panels"] for panel in panels if panel.get("panels"))):
        slots = [(panel["gridPos"]["x"], panel["gridPos"]["y"]) for panel in group]
        assert len(slots) == len(set(slots)), slots


def test_ray_panels_exclude_cumulative_snapshots_and_never_mix_states(rl_run_store) -> None:
    # A forwarded snapshot's `kind` column is always "gauge"; source_temporality carries the real
    # semantics. The Ray allowlist includes cumulative counters, and averaging one in is silently
    # wrong. The spill states are distinct quantities and stay distinct series.
    rows = rl_run_store.execute(_panel_sql("Ray spill bytes")).fetchall()

    by_state = {row[1]: row[2] for row in rows}
    assert by_state == {"Spilled": pytest.approx(2.0e9), "Restored": pytest.approx(5.0e8)}


def test_every_labelled_series_panel_names_the_series_without_its_column() -> None:
    for panel in dashboard_panels(_dashboard()["panels"]):
        if panel.get("type") != "timeseries":
            continue
        if "series" not in {column["selector"] for target in panel.get("targets", []) for column in target["columns"]}:
            continue
        assert panel["fieldConfig"]["defaults"].get("displayName") == "${__field.labels.series}", panel["title"]


@pytest.mark.parametrize("home_minutes", [60, 7 * 24 * 60])
def test_a_listed_run_opens_the_dashboard_whose_picker_offers_it(rl_run_store, home_minutes) -> None:
    home_end_ms = _WINDOW_START_MS + home_minutes * 60000
    rows = []
    for run, kind, minutes, step, name in [
        (ASYNC_RUN_ID, "async", 0, 0, "policy_step"),
        (ASYNC_RUN_ID, "async", 5, 1, "policy_step"),
        (UNSTAMPED_RUN_ID, None, 0, 0, "policy_step"),
        (UNSTAMPED_RUN_ID, "async", -1, 0, "lifecycle"),
        ("mixed-run", None, 0, 0, "policy_step"),
        ("mixed-run", "async", 1, 0, "lifecycle"),
        ("early-stamp-run", "async", 1, 0, "lifecycle"),
        ("early-stamp-run", None, 5, 0, "policy_step"),
        ("end-boundary-run", None, 0, 0, "lifecycle"),
        ("end-boundary-run", None, home_minutes - 0.5, 0, "policy_step"),
        ("end-boundary-run", "async", home_minutes, 0, "lifecycle"),
        (ASYNC_RUN_ID, None, 2, 0, "rollout_call"),
    ]:
        rows.append(
            _row(
                service="marinskyrl",
                name=name,
                value=step,
                moment=WINDOW_START + timedelta(minutes=minutes),
                seq=minutes,
                run_id=run,
                job_id=JOB_ID,
                node_name=NODES[0],
                role="trainer",
                training_type=kind,
            )
        )
    if home_minutes > 120:
        # A run longer than an hour: its link must open the board over the run's last
        # hour — the widest range where the engine panels load on large runs — rather
        # than the whole run.
        rows.extend(
            _row(
                service="marinskyrl",
                name=name,
                value=step,
                moment=WINDOW_START + timedelta(minutes=minutes),
                seq=minutes,
                run_id="long-run",
                job_id=JOB_ID,
                node_name=NODES[0],
                role="trainer",
                training_type=kind,
            )
            for run, kind, minutes, step, name in [
                ("long-run", "sync", home_minutes - 180, 0, "lifecycle"),
                ("long-run", None, home_minutes - 0.5, 2, "policy_step"),
            ]
        )
    rl_run_store.executemany(f'INSERT INTO "telemetry_v1.marinskyrl" VALUES ({", ".join("?" for _ in _COLUMNS)})', rows)
    expected = [
        {
            "section": "recent",
            "run": run,
            "cluster": CLUSTER,
            "step": step,
            "attempts": 1,
            "window_from_ms": _WINDOW_START_MS + first * 60000,
            "window_to_ms": _WINDOW_START_MS + last * 60000,
            "last seen": _WINDOW_START_MS + minutes * 60000,
            "type": kind,
        }
        for run, step, minutes, first, last, kind in [
            (RUN_ID, 5, 25, 0, 26, "sync"),
            (ASYNC_RUN_ID, 1, 5, 0, 6, "async"),
            (UNSTAMPED_RUN_ID, 0, 0, 0, 1, "sync"),
            ("mixed-run", 0, 0, 0, 2, "async"),
            ("early-stamp-run", 0, 5, 0, 6, "async"),
            ("end-boundary-run", 0, home_minutes - 0.5, max(0, home_minutes - 60), home_minutes, "sync"),
        ]
    ]
    if home_minutes > 120:
        # The clamp opens a >1h run's link over its last hour: window_from at
        # window_to - RL_RECENT_RUN_LINK_WINDOW_MS, window_to at the run's end.
        expected.append(
            {
                "section": "recent",
                "run": "long-run",
                "cluster": CLUSTER,
                "step": 2,
                "attempts": 1,
                "window_from_ms": _WINDOW_START_MS + (home_minutes - 60) * 60000,
                "window_to_ms": _WINDOW_START_MS + home_minutes * 60000,
                "last seen": _WINDOW_START_MS + (home_minutes - 0.5) * 60000,
                "type": "sync",
            }
        )
    actual = _request_rows(
        rl_run_store, "/v1/rl/recent", {"from": _WINDOW_START_MS, "to": home_end_ms, "view": "recent"}
    )
    assert sorted(actual, key=lambda row: row["run"]) == sorted(expected, key=lambda row: row["run"])
    (links,) = [
        prop["value"]
        for override in _recent_runs_panel()["fieldConfig"]["overrides"]
        if override["matcher"]["options"] == "run"
        for prop in override["properties"]
        if prop["id"] == "links"
    ]
    template = links[0]["url"]
    run_variable = next(v for v in _dashboard()["templating"]["list"] if v["name"] == "run")
    query = run_variable["query"]["infinityQuery"]
    linked_options = {
        RUN_ID: [RUN_ID, UNSTAMPED_RUN_ID, "end-boundary-run"],
        ASYNC_RUN_ID: [ASYNC_RUN_ID, "early-stamp-run", "mixed-run"],
        UNSTAMPED_RUN_ID: [RUN_ID, UNSTAMPED_RUN_ID, "end-boundary-run", "mixed-run"],
        "mixed-run": [ASYNC_RUN_ID, "early-stamp-run", "mixed-run"],
        "early-stamp-run": [ASYNC_RUN_ID, "early-stamp-run", "mixed-run"],
        "end-boundary-run": [RUN_ID, UNSTAMPED_RUN_ID, "end-boundary-run"],
    }
    if home_minutes > 120:
        # Runs clamped to the board's last hour open a picker window that only sees
        # the runs with records in that hour.
        linked_options["end-boundary-run"] = ["end-boundary-run", "long-run"]
        linked_options["long-run"] = ["end-boundary-run", "long-run"]
    for row in actual:
        concrete = re.sub(r"\$\{__data.fields.([^}]+)\}", lambda match, row=row: str(row[match[1]]), template)
        url = urlparse(concrete)
        assert url.path == "/d/marin-rl-runs"
        assert parse_qs(url.query) == {
            "var-training_type": [row["type"]],
            "var-run": [row["run"]],
            "var-cluster": [CLUSTER],
            "from": [str(row["window_from_ms"])],
            "to": [str(row["window_to_ms"])],
        }
        linked = parse_qs(url.query)
        params = resolved_query_params(
            query,
            (
                ("${cluster:sqlstring}", f"'{linked['var-cluster'][0]}'"),
                ("${training_type:sqlstring}", f"'{linked['var-training_type'][0]}'"),
                ("${__from}", linked["from"][0]),
                ("${__to}", linked["to"][0]),
            ),
        )
        offered = _request_rows(rl_run_store, query["url"], params)
        assert sorted(offered, key=lambda item: item["value"]) == [
            {"value": run} for run in sorted(linked_options[row["run"]])
        ]
        if row["run"] == "end-boundary-run":
            assert (
                _request_rows(
                    rl_run_store,
                    "/v1/rl/overview",
                    {
                        "clusters": CLUSTER,
                        "run": linked["var-run"][0],
                        "from": linked["from"][0],
                        "to": linked["to"][0],
                        "bucket_ms": 60000,
                        "view": "critical_path",
                    },
                )
                == []
            )
    for kind, wanted in (
        (
            "sync",
            [RUN_ID, UNSTAMPED_RUN_ID, "end-boundary-run"] + (["long-run"] if home_minutes > 120 else []),
        ),
        ("async", [ASYNC_RUN_ID, "early-stamp-run", "mixed-run"]),
    ):
        params = resolved_query_params(
            query,
            (
                ("${cluster:sqlstring}", f"'{CLUSTER}'"),
                ("${training_type:sqlstring}", f"'{kind}'"),
                ("${__from}", str(_WINDOW_START_MS)),
                ("${__to}", str(home_end_ms)),
            ),
        )
        offered = _request_rows(rl_run_store, query["url"], params)
        assert sorted(offered, key=lambda row: row["value"]) == [{"value": run} for run in sorted(wanted)]


def test_the_job_and_execution_pickers_offer_what_the_producers_report(rl_run_store) -> None:
    other_job = "/atqamar/snowball-e6-muonh-0-attempt-1"
    quiet_execution = "iris:/atqamar/snowball-e6-muonh-0-attempt-0/9:attempt:9"
    rows = [
        _row(
            service="marinskyrl",
            name="phase_duration_seconds",
            value=1.0,
            moment=WINDOW_START + timedelta(minutes=5 * bucket),
            seq=bucket,
            run_id=RUN_ID,
            job_id=job_id,
            execution_uid=execution,
            attributes={"phase": "train_step", "clock_domain": "critical_path", "outcome": "success"},
        )
        for bucket in range(2)
        for job_id, execution in ((other_job, f"iris:{other_job}/0:attempt:0"), (JOB_ID, quiet_execution))
    ]
    rl_run_store.executemany(f'INSERT INTO "telemetry_v1.marinskyrl" VALUES ({", ".join("?" for _ in _COLUMNS)})', rows)

    variables = {variable["name"]: variable for variable in _dashboard()["templating"]["list"]}
    substitutions = (
        ("${cluster:sqlstring}", f"'{CLUSTER}'"),
        ("${run:sqlstring}", f"'{RUN_ID}'"),
        ("${job:sqlstring}", f"'{JOB_ID}'"),
        ("${__from}", str(_WINDOW_START_MS)),
        ("${__to}", str(_NOW_MS)),
    )

    def picker_rows(name: str) -> list[dict]:
        query = variables[name]["query"]["infinityQuery"]
        return _request_rows(rl_run_store, query["url"], resolved_query_params(query, substitutions))

    assert picker_rows("job") == [{"value": JOB_ID}]
    assert picker_rows("execution") == [{"value": DEFAULT_EXECUTION_UID}]
