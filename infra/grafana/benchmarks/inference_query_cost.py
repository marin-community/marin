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
from pathlib import Path

import pyarrow.ipc as ipc
import server
from config import BridgeConfig, ClusterTarget
from dashboard_stitch import stitch_all
from finelog.deploy.config import load_finelog_config
from finelog.deploy.connect import open_client
from starlette.testclient import TestClient


class RecordingSource:
    """Record actual Finelog calls made by the bridge, including shared failures."""

    def __init__(self, client, output: Path):
        self.target = ClusterTarget("marin", "project", "zone", "fleet", "cluster")
        self.client = client
        self.output = output
        self.calls = []
        self.guard = threading.Lock()

    def query(self, sql: str, *, max_rows: int):
        with self.guard:
            index = len(self.calls)
            record = {"index": index}
            self.calls.append(record)
        (self.output / f"query-{index}.sql").write_text(sql)
        started = time.monotonic()
        try:
            table = self.client.query(sql, max_rows=max_rows)
            record.update(seconds=time.monotonic() - started, rows=table.num_rows, bytes=table.nbytes)
            with ipc.new_file(str(self.output / f"query-{index}.arrow"), table.schema) as writer:
                writer.write_table(table)
            return table
        except Exception as error:
            record.update(seconds=time.monotonic() - started, error=str(error))
            raise


def main():
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
    args = parser.parse_args()
    assert (
        Path(server.__file__).resolve().parent == args.grafana_dir.resolve() / "src"
    ), "PYTHONPATH must select this bridge"
    args.output.mkdir(parents=True, exist_ok=True)
    directory = args.grafana_dir / "dashboards"
    dashboards = stitch_all(directory, directory / "panels")
    config = BridgeConfig(
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
    params = {
        "identity_kind": args.identity_kind,
        "identity": args.identity,
        "from": args.from_ms,
        "to": args.to_ms,
        "bucket_ms": args.bucket_ms,
    }
    pages = []
    filenames = ("inference_overview.json", "inference.json")
    if args.first_page == "diagnostics":
        filenames = filenames[::-1]
    with open_client(load_finelog_config("marin"), "marin", tunnel_timeout=30, request_timeout=20) as upstream:
        source = RecordingSource(upstream, args.output)
        app = server.create_app(config, {"marin": source}, {}, None, None, None)
        with TestClient(app, raise_server_exceptions=False) as client:
            phases = ["cold", "warm"] + (["refresh"] if args.refresh_after else [])
            for phase in phases:
                if phase == "refresh":
                    time.sleep(args.refresh_after)
                    params["from"] += args.refresh_after * 1000
                    params["to"] += args.refresh_after * 1000
                for filename in filenames:
                    dashboard = dashboards[filename]
                    started, before = time.monotonic(), len(source.calls)
                    variable = next(v for v in dashboard["templating"]["list"] if v["name"] == "identity")["query"][
                        "infinityQuery"
                    ]
                    substitutions = {
                        "${identity_kind}": args.identity_kind,
                        "${__from}": str(params["from"]),
                        "${__to}": str(params["to"]),
                    }
                    selector_params = {}
                    for param in variable["url_options"]["params"]:
                        value = param["value"]
                        for macro, replacement in substitutions.items():
                            value = value.replace(macro, replacement)
                        selector_params[param["key"]] = value
                    selector = client.get("/finelog/marin/query", params=selector_params)
                    requests = []
                    for panel in dashboard["panels"]:
                        for target in panel.get("targets", []):
                            view = next(p["value"] for p in target["url_options"]["params"] if p["key"] == "view")
                            requests.append((panel["title"], target["url"], view))

                    def fetch(request):
                        title, path, view = request
                        began = time.monotonic()
                        response = client.get("/finelog/marin" + path, params={**params, "view": view})
                        return {
                            "title": title,
                            "view": view,
                            "status": response.status_code,
                            "seconds": time.monotonic() - began,
                            "rows": response.json() if response.status_code == 200 else response.text,
                        }

                    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
                        panels = list(pool.map(fetch, requests))
                    result = {
                        "phase": phase,
                        "dashboard": filename,
                        "params": dict(params),
                        "seconds": time.monotonic() - started,
                        "query_count": len(source.calls) - before,
                        "selector_status": selector.status_code,
                        "selector": selector.json() if selector.status_code == 200 else selector.text,
                        "panels": panels,
                    }
                    pages.append(result)
                    (args.output / "result.json").write_text(
                        json.dumps({"queries": source.calls, "pages": pages}, indent=2)
                    )
                    print(json.dumps({k: v for k, v in result.items() if k not in ("panels", "selector")}), flush=True)


if __name__ == "__main__":
    main()
