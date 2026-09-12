# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run local shuffles, persist diagnostics in Finelog, and render their histograms."""

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode

from finelog.client import LogClient
from finelog.embedded import EmbeddedServer
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from rigging.connect import IapAuth
from rigging.credentials import iap_provider_for
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.shuffle_report import render_shuffle_report
from zephyr.stats import StatsConfig, ZephyrShuffleStat

NUM_OUTPUT_SHARDS = 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-url", help="Explicit Finelog URL; otherwise start an embedded local server.")
    parser.add_argument("--auth-profile", help="Marin IAP credential profile, e.g. marin.")
    parser.add_argument("--output", type=Path, default=Path("/tmp/zephyr-shuffle-report.html"))
    args = parser.parse_args()
    if args.auth_profile and not args.stats_url:
        parser.error("--auth-profile requires --stats-url")

    with ExitStack() as stack:
        scratch = Path(stack.enter_context(TemporaryDirectory(prefix="zephyr-shuffle-demo-")))
        url = args.stats_url
        if url is None:
            server = EmbeddedServer(log_dir=str(scratch / "finelog"))
            stack.callback(server.stop)
            url = f"http://127.0.0.1:{server.port}"
        client = LocalClient()
        stack.callback(client.shutdown)
        executions = []
        with ZephyrContext(
            client=client,
            max_workers=2,
            resources=ResourceConfig(cpu=1, ram="1g"),
            chunk_storage_prefix=str(scratch / "chunks"),
            stats_config=StatsConfig(url, args.auth_profile),
        ) as context:
            for scenario in ("uniform", "hot-key"):
                keys = [0 if scenario == "hot-key" and index < 9000 else index % 128 for index in range(10_000)]
                rows = [{"key": key, "count": 1} for key in keys]
                dataset = (
                    Dataset.from_list(rows)
                    .reshard(8)
                    .group_by(
                        key=lambda row: row["key"],
                        reducer=lambda key, items: (key, sum(row["count"] for row in items)),
                        num_output_shards=NUM_OUTPUT_SHARDS,
                    )
                )
                result = context.execute(dataset)
                assert dict(result.results) == dict(Counter(keys))
                executions.append(result.execution_id)
                print(f"{scenario}: {result.execution_id}")

        interceptors = IapAuth(iap_provider_for(args.auth_profile)).interceptors() if args.auth_profile else ()
        query_client = LogClient.connect(url, interceptors=interceptors)
        stack.callback(query_client.close)
        allowed_fields = {field.name for field in fields(ZephyrShuffleStat)}
        records = []
        for execution_id in executions:
            quoted_id = execution_id.replace("'", "''")
            result_rows = query_client.query(
                "WITH reports AS (SELECT *, ROW_NUMBER() OVER ("
                "PARTITION BY execution_id, stage_name, target_shard "
                "ORDER BY attempt DESC, (input_rows IS NOT NULL) DESC, ts DESC, seq DESC) AS report_rank "
                f"FROM \"zephyr.shuffle\" WHERE execution_id = '{quoted_id}' "
                "AND ts >= now() - INTERVAL '1 hour' AND ts <= now()) "
                "SELECT * FROM reports WHERE report_rank = 1 ORDER BY target_shard"
            ).to_pylist()
            assert (
                len(result_rows) == NUM_OUTPUT_SHARDS
            ), f"Expected {NUM_OUTPUT_SHARDS} persisted targets, got {len(result_rows)}"
            assert {row["num_targets"] for row in result_rows} == {NUM_OUTPUT_SHARDS}
            assert all(row["input_rows"] is not None for row in result_rows), "Some targets remain unreported"
            assert sum(row["input_rows"] for row in result_rows) == 10_000
            records.extend(
                ZephyrShuffleStat(**{key: value for key, value in row.items() if key in allowed_fields})
                for row in result_rows
            )
            parameters = urlencode({"var-execution_id": execution_id, "from": "now-1h", "to": "now"})
            print(f"Grafana (after dashboard deployment): https://grafana.oa.dev/d/marin-zephyr-shuffle?{parameters}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        render_shuffle_report(records, args.output)
        print(f"Visual report from {len(records)} persisted rows: {args.output.resolve()}")


if __name__ == "__main__":
    main()
