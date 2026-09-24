# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay both inference pages against Finelog, saving SQL, Arrow inputs, and responses.

Run from the repository root with PYTHONPATH=infra/grafana/src and uv run python.
To measure an older bridge, set PYTHONPATH to its src/ and pass its --grafana-dir.
The output directory should be scratch storage, not the worktree.
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.ipc as ipc
import server
from config import BridgeConfig, ClusterTarget
from dashboard_stitch import stitch_all
from finelog.deploy.config import load_finelog_config
from finelog.deploy.connect import open_client
from starlette.testclient import TestClient

FINELOG_CLUSTER = "marin"
INFERENCE_DASHBOARDS = ("inference_overview.json", "inference.json")


@dataclass
class QueryRecord:
    """One Finelog call made by the replay."""

    index: int
    seconds: float | None = None
    rows: int | None = None
    bytes: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class PanelRequest:
    title: str
    path: str
    view: str
    bucket_ms: str


class RecordingSource:
    """Record actual Finelog calls made by the bridge, including shared failures."""

    def __init__(self, client, output: Path):
        self.target = ClusterTarget(FINELOG_CLUSTER, "project", "zone", "fleet", "cluster")
        self.client = client
        self.output = output
        self.calls: list[QueryRecord] = []
        self.guard = threading.Lock()

    def query(self, sql: str, *, max_rows: int):
        with self.guard:
            record = QueryRecord(index=len(self.calls))
            self.calls.append(record)
        (self.output / f"query-{record.index}.sql").write_text(sql)
        started = time.monotonic()
        try:
            table = self.client.query(sql, max_rows=max_rows)
            record.seconds = time.monotonic() - started
            record.rows = table.num_rows
            record.bytes = table.nbytes
            with ipc.new_file(str(self.output / f"query-{record.index}.arrow"), table.schema) as writer:
                writer.write_table(table)
            return table
        except Exception as error:
            record.seconds = time.monotonic() - started
            record.error = str(error)
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grafana-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--identity", required=True)
    parser.add_argument("--identity-kind", default="run_id", choices=["job_id", "run_id", "execution_uid"])
    parser.add_argument("--from-ms", type=int, required=True)
    parser.add_argument("--to-ms", type=int, required=True)
    parser.add_argument("--bucket-ms", type=int, default=15_000)
    parser.add_argument("--first-page", choices=["overview", "diagnostics"], default="overview")
    parser.add_argument(
        "--refresh-after", type=int, default=0, help="Wait this many seconds, then advance the window and refresh"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def replay_bridge_config() -> BridgeConfig:
    return BridgeConfig(
        max_rows=10_000,
        cache_ttl=20,
        query_timeout_ms=20_000,
        iris_cache_ttl=15,
        github_cache_ttl=60,
        k8s_cache_ttl=30,
        http_timeout=5,
        github_app_credentials=None,
        cw_read_token=None,
        loom_alerts=None,
    )


def selector_params(dashboard: dict, params: dict[str, str | int]) -> dict[str, str]:
    variable = next(item for item in dashboard["templating"]["list"] if item["name"] == "identity")["query"][
        "infinityQuery"
    ]
    substitutions = {
        "${identity_kind}": str(params["identity_kind"]),
        "${__from}": str(params["from"]),
        "${__to}": str(params["to"]),
    }
    result = {}
    for param in variable["url_options"]["params"]:
        value = param["value"]
        for macro, replacement in substitutions.items():
            value = value.replace(macro, replacement)
        result[param["key"]] = value
    return result


def panel_requests(dashboard: dict) -> list[PanelRequest]:
    requests = []
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            target_params = {param["key"]: param["value"] for param in target["url_options"]["params"]}
            requests.append(
                PanelRequest(panel["title"], target["url"], target_params["view"], target_params["bucket_ms"])
            )
    return requests


def fetch_panel(client: TestClient, params: dict[str, str | int], request: PanelRequest) -> dict:
    bucket_ms = int(params["bucket_ms"]) if request.bucket_ms == "${__interval_ms}" else int(request.bucket_ms)
    started = time.monotonic()
    response = client.get(
        f"/finelog/{FINELOG_CLUSTER}{request.path}", params={**params, "view": request.view, "bucket_ms": bucket_ms}
    )
    return {
        "title": request.title,
        "view": request.view,
        "bucket_ms": bucket_ms,
        "status": response.status_code,
        "seconds": time.monotonic() - started,
        "rows": response.json() if response.status_code == 200 else response.text,
    }


def replay_page(
    client: TestClient,
    source: RecordingSource,
    dashboard: dict,
    filename: str,
    phase: str,
    params: dict[str, str | int],
) -> dict:
    started, before = time.monotonic(), len(source.calls)
    selector = client.get(f"/finelog/{FINELOG_CLUSTER}/query", params=selector_params(dashboard, params))
    requests = panel_requests(dashboard)
    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        panels = list(pool.map(lambda request: fetch_panel(client, params, request), requests))
    return {
        "phase": phase,
        "dashboard": filename,
        "params": dict(params),
        "seconds": time.monotonic() - started,
        "query_count": len(source.calls) - before,
        "selector_status": selector.status_code,
        "selector": selector.json() if selector.status_code == 200 else selector.text,
        "panels": panels,
    }


def run_replay(args: argparse.Namespace) -> None:
    assert (
        Path(server.__file__).resolve().parent == args.grafana_dir.resolve() / "src"
    ), "PYTHONPATH must select this bridge"
    args.output.mkdir(parents=True, exist_ok=True)
    dashboard_dir = args.grafana_dir / "dashboards"
    dashboards = stitch_all(dashboard_dir, dashboard_dir / "panels")
    params = {
        "identity_kind": args.identity_kind,
        "identity": args.identity,
        "from": args.from_ms,
        "to": args.to_ms,
        "bucket_ms": args.bucket_ms,
    }
    filenames = INFERENCE_DASHBOARDS if args.first_page == "overview" else INFERENCE_DASHBOARDS[::-1]
    pages = []
    config = load_finelog_config(FINELOG_CLUSTER)
    with open_client(config, FINELOG_CLUSTER, tunnel_timeout=30, request_timeout=20) as upstream:
        source = RecordingSource(upstream, args.output)
        app = server.create_app(replay_bridge_config(), {FINELOG_CLUSTER: source}, {}, None, None, None)
        with TestClient(app, raise_server_exceptions=False) as client:
            phases = ["cold", "warm"] + (["refresh"] if args.refresh_after else [])
            for phase in phases:
                if phase == "refresh":
                    time.sleep(args.refresh_after)
                    params["from"] += args.refresh_after * 1000
                    params["to"] += args.refresh_after * 1000
                for filename in filenames:
                    result = replay_page(client, source, dashboards[filename], filename, phase, params)
                    pages.append(result)
                    payload = {"queries": [asdict(record) for record in source.calls], "pages": pages}
                    (args.output / "result.json").write_text(json.dumps(payload, indent=2))
                    print(json.dumps({key: value for key, value in result.items() if key not in ("panels", "selector")}))


def main() -> None:
    run_replay(parse_args())


if __name__ == "__main__":
    main()
