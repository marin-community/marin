# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay visible Finelog dashboard targets against an existing server.

Run with PYTHONPATH=<selected-grafana-src>:infra/grafana and
``uv run --directory infra/grafana --no-sync python -m benchmarks.dashboard_query_cost``.
Use --grafana-dir for the selected checkout and --variables for a JSON object
mapping dashboard filenames to variable names and string or string-list values.
Selectors and external data sources are excluded. Unspecified maxDataPoints uses
1,000 points; panel intervals are lower bounds on Grafana's rounded interval.

--live-window freezes only the bridge module's wall clock to one second after
--to-ms, exercising live alignment on settled historical data. Each timeFrom
override has one millisecond of additional clock skew per panel. Other module
clocks and all monotonic timers remain real. Each dashboard iteration starts
with empty bridge caches; --warm repeats the page against those same caches.
SQL, Arrow inputs, responses, and timing summaries are retained under --output.
Query counts and byte totals cover SQL calls; namespace discovery RPCs are
reported separately as namespace_calls.
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import server
from dashboard_stitch import stitch_all
from finelog.client import LogClient
from starlette.testclient import TestClient

from .inference_query_cost import FINELOG_CLUSTER, RecordingSource, replay_bridge_config

_DURATION_UNITS = {"ms": 1, "s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
_INTERVAL_STEPS = (
    1,
    10,
    20,
    50,
    100,
    200,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    15_000,
    20_000,
    30_000,
    60_000,
    120_000,
    300_000,
    600_000,
    900_000,
    1_200_000,
    1_800_000,
    3_600_000,
    7_200_000,
    10_800_000,
    21_600_000,
    43_200_000,
    86_400_000,
    604_800_000,
    2_592_000_000,
    31_536_000_000,
)
_VARIABLE = re.compile(r"\$\{([^}]+)\}")


@dataclass(frozen=True)
class PanelRequest:
    panel_id: int
    title: str
    ref_id: str
    path: str
    params: dict[str, str]
    from_ms: int
    to_ms: int
    interval_ms: int


def duration_ms(value: str) -> int:
    """Parse an unrounded Grafana relative duration."""
    match = re.fullmatch(r"(?:now-)?(\d+)(ms|s|m|h|d|w)", value)
    if match is None:
        raise ValueError(f"unsupported Grafana duration {value!r}")
    return int(match[1]) * _DURATION_UNITS[match[2]]


def interval_ms(duration: int, max_points: int, minimum: str = "") -> int:
    """Round to Grafana's interval steps and apply the panel minimum."""
    raw = duration / max_points
    rounded = _INTERVAL_STEPS[-1]
    for current, following in pairwise(_INTERVAL_STEPS):
        if raw < (current + following) / 2:
            rounded = current
            break
    return max(rounded, duration_ms(minimum.removeprefix(">")) if minimum else 1)


def substitute(value: str, variables: dict[str, str | list[str]], macros: dict[str, int]) -> str:
    """Interpolate supported Grafana variables, rejecting unresolved macros."""

    def replacement(match: re.Match[str]) -> str:
        expression = match[1]
        if expression in macros:
            return str(macros[expression])
        name, _, formatting = expression.partition(":")
        if name not in variables or formatting not in ("", "csv", "sqlstring"):
            raise ValueError(f"unsupported or missing Grafana variable {match[0]!r}")
        configured = variables[name]
        values = [configured] if isinstance(configured, str) else configured
        if formatting == "sqlstring":
            return ",".join("'" + item.replace("'", "''") + "'" for item in values)
        return ",".join(values)

    rendered = _VARIABLE.sub(replacement, value)
    if re.search(r"\$[A-Za-z_{]", rendered):
        raise ValueError(f"unsupported Grafana macro in {rendered!r}")
    return rendered


def panel_requests(
    dashboard: dict,
    variables: dict[str, str | list[str]],
    end_ms: int,
) -> tuple[list[PanelRequest], list[dict]]:
    """Resolve visible Finelog targets and explain every skipped target or row."""
    if dashboard["time"]["to"] != "now":
        raise ValueError("only dashboard ranges ending at now are supported")
    default_duration = duration_ms(dashboard["time"]["from"])
    requests: list[PanelRequest] = []
    skipped: list[dict] = []
    panel_number = 0

    def visit(panels: list[dict]) -> None:
        nonlocal panel_number
        for panel in panels:
            panel_number += 1
            identity = {"panel_id": panel["id"], "title": panel.get("title", "")}
            if panel.get("type") == "row":
                if panel.get("collapsed"):
                    skipped.append({**identity, "reason": "collapsed row"})
                else:
                    visit(panel.get("panels", []))
                continue
            override = panel.get("timeFrom")
            duration = duration_ms(override) if override else default_duration
            panel_end = end_ms + (panel_number if override else 0)
            panel_start = panel_end - duration
            bucket = interval_ms(duration, panel.get("maxDataPoints", 1_000), panel.get("interval", ""))
            macros = {"__from": panel_start, "__to": panel_end, "__interval_ms": bucket}
            for target in panel.get("targets", []):
                target_identity = {**identity, "ref_id": target.get("refId", "")}
                if target.get("hide"):
                    skipped.append({**target_identity, "reason": "hidden target"})
                    continue
                path = target.get("url", "")
                datasource = target.get("datasource", panel.get("datasource", {}))
                uid = datasource.get("uid") if isinstance(datasource, dict) else datasource
                if path.startswith("/finelog/marin/"):
                    pass
                elif uid == "finelog-marin":
                    if path in ("/fleet_health", "/alerts/fleet_health"):
                        skipped.append({**target_identity, "reason": "external Kubernetes health probes"})
                        continue
                    path = "/finelog/marin" + path
                else:
                    skipped.append({**target_identity, "reason": f"external datasource {uid!r}"})
                    continue
                options = target.get("url_options", {})
                if options.get("method", "GET") != "GET":
                    raise ValueError(f"unsupported target method for {target_identity}")
                params = {
                    item["key"]: substitute(item["value"], variables, macros) for item in options.get("params", [])
                }
                requests.append(
                    PanelRequest(
                        panel["id"],
                        panel.get("title", ""),
                        target.get("refId", ""),
                        substitute(path, variables, macros),
                        params,
                        panel_start,
                        panel_end,
                        bucket,
                    )
                )
            if not panel.get("targets"):
                skipped.append({**identity, "reason": "no query targets"})

    visit(dashboard["panels"])
    return requests, skipped


def fetch_panel(client: TestClient, request: PanelRequest) -> dict:
    started = time.monotonic()
    response = client.get(request.path, params=request.params)
    elapsed = time.monotonic() - started
    body = response.json() if "application/json" in response.headers.get("content-type", "") else response.text
    return {
        **asdict(request),
        "seconds": elapsed,
        "status": response.status_code,
        "response_bytes": len(response.content),
        "response": body,
    }


def replay_page(client: TestClient, source: RecordingSource, requests: list[PanelRequest], workers: int) -> dict:
    """Load the page concurrently and summarize actual upstream work."""
    before = len(source.calls)
    before_namespaces = source.namespace_calls
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        panels = list(pool.map(lambda request: fetch_panel(client, request), requests))
    elapsed = time.monotonic() - started
    queries = source.calls[before:]
    return {
        "seconds": elapsed,
        "query_count": len(queries),
        "namespace_calls": source.namespace_calls - before_namespaces,
        "query_rows": sum(query.rows or 0 for query in queries),
        "query_bytes": sum(query.bytes or 0 for query in queries),
        "query_seconds": sum(query.seconds or 0 for query in queries),
        "queries": [asdict(query) for query in queries],
        "panels": panels,
    }


def run_replay(args: argparse.Namespace) -> dict:
    """Replay selected dashboards with a fresh bridge cache per iteration."""
    if Path(server.__file__).resolve().parent != args.grafana_dir.resolve() / "src":
        raise ValueError("PYTHONPATH must select the --grafana-dir bridge src directory")
    dashboard_dir = args.grafana_dir / "dashboards"
    dashboards = stitch_all(dashboard_dir, dashboard_dir / "panels")
    configured = json.loads(args.variables.read_text())
    filenames = args.dashboard or sorted(dashboards)
    prepared = {
        filename: panel_requests(dashboards[filename], configured.get(filename, {}), args.to_ms)
        for filename in filenames
    }
    args.output.mkdir(parents=True, exist_ok=False)
    original_time = server.time
    if args.live_window:
        server.time = SimpleNamespace(time=lambda: args.to_ms / 1_000 + 1, monotonic=time.monotonic)
    result = {
        "grafana_dir": str(args.grafana_dir.resolve()),
        "address": args.address,
        "to_ms": args.to_ms,
        "live_window": args.live_window,
        "workers": args.workers,
        "variables": configured,
        "pages": [],
    }
    try:
        with closing(LogClient.connect(args.address, timeout_ms=60_000)) as upstream:
            for iteration in range(args.iterations):
                for filename, (requests, skipped) in prepared.items():
                    output = args.output / Path(filename).stem / f"iteration-{iteration}"
                    output.mkdir(parents=True)
                    source = RecordingSource(upstream, output)
                    config = replace(replay_bridge_config(), max_rows=500_000, query_timeout_ms=60_000)
                    app = server.create_app(config, {FINELOG_CLUSTER: source}, {}, None, None, None)
                    with TestClient(app, raise_server_exceptions=False) as client:
                        for phase in (("cold", "warm") if args.warm else ("cold",)):
                            page = {
                                "dashboard": filename,
                                "iteration": iteration,
                                "phase": phase,
                                "skipped_panels": skipped,
                                **replay_page(client, source, requests, args.workers),
                            }
                            result["pages"].append(page)
                            (output / f"{phase}.json").write_text(json.dumps(page, indent=2) + "\n")
                            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                            print(
                                json.dumps(
                                    {
                                        key: value
                                        for key, value in page.items()
                                        if key not in ("panels", "queries", "skipped_panels")
                                    }
                                ),
                                flush=True,
                            )
    finally:
        server.time = original_time
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grafana-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--dashboard", action="append", help="Dashboard filename; repeat to select several (default: all)"
    )
    parser.add_argument("--address", required=True, help="Existing Finelog server or loopback tunnel")
    parser.add_argument("--to-ms", type=int, required=True, help="Fixed settled-data window end")
    parser.add_argument("--variables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--live-window", action="store_true")
    parser.add_argument("--warm", action="store_true", help="Repeat each page with the same bridge caches")
    result = run_replay(parser.parse_args())
    failures = [
        f"{page['dashboard']} panel {panel['panel_id']} ({panel['title']}): HTTP {panel['status']}"
        for page in result["pages"]
        for panel in page["panels"]
        if not 200 <= panel["status"] < 300
    ]
    if failures:
        parser.exit(1, "\n".join(failures) + "\n")


if __name__ == "__main__":
    main()
