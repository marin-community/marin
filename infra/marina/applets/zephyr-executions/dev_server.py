# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the built applet on loopback against an embedded Finelog seeded by real pipelines.

Run from the repository root::

    uv run python infra/marina/applets/zephyr-executions/dev_server.py --port 3104

Three local executions are seeded: a uniform two-shuffle run, a hot-key run,
and a join. A fourth plan record dated 30 days ago sits outside the listing
window, so its deep link exercises the direct lookup. The page is served at
``/`` and the backend at ``/api/``, the same relative layout Marina uses. Not
part of the published package.
"""

import argparse
import logging
import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from finelog.client import LogClient
from finelog.embedded import EmbeddedServer
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from server.app import create_api
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.stats import ZEPHYR_EXECUTION_STATS_NAMESPACE, StatsConfig, ZephyrExecutionStat

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("dev_server")


def seed(url: str, scratch: Path) -> list[str]:
    def run(dataset: Dataset, name: str) -> str:
        with ZephyrContext(
            client=LocalClient(),
            max_workers=2,
            resources=ResourceConfig(cpu=1, ram="512m"),
            chunk_storage_prefix=str(scratch / name),
            stats_config=StatsConfig(url),
        ) as context:
            result = context.execute(dataset)
        logger.info("seeded %s as %s", name, result.execution_id)
        return result.execution_id

    def key_sum(key, rows):
        return (key, sum(row["value"] for row in rows))

    uniform = (
        Dataset.from_list([{"key": index % 128, "value": index} for index in range(10_000)])
        .reshard(8)
        .group_by(key=lambda row: row["key"], reducer=key_sum, num_output_shards=64)
        .group_by(
            key=lambda pair: pair[0] % 8,
            reducer=lambda key, pairs: (key, sum(total for _, total in pairs)),
            num_output_shards=8,
        )
    )
    hot = (
        Dataset.from_list([{"key": 0 if index % 10 else index % 128, "value": index} for index in range(10_000)])
        .reshard(8)
        .group_by(key=lambda row: row["key"], reducer=key_sum, num_output_shards=16)
    )
    left = Dataset.from_list([{"id": index, "text": f"row {index}"} for index in range(50)]).group_by(
        key=lambda row: row["id"], reducer=lambda key, rows: next(rows), num_output_shards=4
    )
    right = Dataset.from_list([{"id": index, "score": index * 7} for index in range(0, 50, 3)]).group_by(
        key=lambda row: row["id"], reducer=lambda key, rows: next(rows), num_output_shards=4
    )
    joined = left.sorted_merge_join(right, left_key=lambda row: row["id"], right_key=lambda row: row["id"])
    executions = [run(uniform, "uniform"), run(hot, "hot-key"), run(joined, "join")]
    executions.append(seed_old_plan(url, executions[0]))
    return executions


def seed_old_plan(url: str, template_id: str) -> str:
    """Copy the first execution's plan record under a new id dated 30 days ago."""
    client = LogClient.connect(url)
    try:
        rows = client.query(
            f"""SELECT * FROM "{ZEPHYR_EXECUTION_STATS_NAMESPACE}" WHERE execution_id = '{template_id}'"""
        ).to_pylist()
        old = ZephyrExecutionStat(
            execution_id="20260816-120000-0ld1d0c5",
            root_job_id="/karan/old-run",
            coordinator_job_id="/karan/old-run/coordinator",
            ts=datetime.now(UTC) - timedelta(days=30),
            input_shards=rows[0]["input_shards"],
            stages_json=rows[0]["stages_json"],
        )
        table = client.get_table(ZEPHYR_EXECUTION_STATS_NAMESPACE, ZephyrExecutionStat)
        table.write([old])
        assert table.flush(timeout=30) is not None
    finally:
        client.close()
    return old.execution_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=3104)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    dist = HERE / "dist"
    if not (dist / "index.html").exists():
        sys.exit("dist/index.html is missing; run `npm run build` in the applet directory first")

    scratch = Path(tempfile.mkdtemp(prefix="zephyr-applet-dev-"))
    finelog = EmbeddedServer(log_dir=str(scratch / "finelog"))
    url = f"http://127.0.0.1:{finelog.port}"
    executions = seed(url, scratch)
    os.environ["ZEPHYR_APPLET_FINELOG_URL"] = url
    os.environ.pop("ZEPHYR_APPLET_IAP_CLUSTER", None)

    app = FastAPI()
    app.mount("/api", create_api(None))
    app.mount("/", StaticFiles(directory=str(dist), html=True), name="dist")
    for execution_id in executions:
        print(f"http://127.0.0.1:{args.port}/#/execution/{execution_id}")
    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        finelog.stop()


if __name__ == "__main__":
    main()
